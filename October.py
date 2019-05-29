from pyobb.obb import OBB
# According to the author of PYOBB:
# local frame: rotated but no translation
# all coords: under this local frame
import numpy as np
# October: Octree + OBB
class October:
    def __init__(self, points, spv_thres, upper_cent=None):
        self.singleton=(len(points)==1)
        self.upper_cent=upper_cent
        self.points=points
        self.originalcenter=np.mean(points)
        # Singleton: the simplest case
        if self.singleton:
            self.obb=None
            self.children=None
            return
        # Interpolate a middle point if there are only two
        if len(self.points)==2:
            p0=self.points[0]
            p1=self.points[1]
            self.points.append(list(map(sum,zip(p0,p1))))
        self.obb=OBB.build_from_points(points)
        # Stop splitting when necessary
        if self.spv()<spv_thres or len(self.points)<=100:
            # print("#Points:"+str(len(self.points)))
            # print("SPV:"+str(self.spv()))
            self.children=None
        else:
            p=self.splitpoints()
            self.children=[None]*8
            for i in range(8):
                if p[i]!=[]:
                    if len(p[i])==1:
                        self.children[i]=October(p[i],spv_thres,upper_cent=self.originalcenter)
                    else:
                        self.children[i]=October(p[i],spv_thres)

    def spv(self):
        l=self.obb.max-self.obb.min
        return l[0]*l[1]*l[2]/len(self.points)

    def splitpoints(self):
        newpoints=self.transform(self.points)
        ps={}
        for i in range(8):
            ps[i]=[]
        for i in range(len(newpoints)):
            ps[self.getidx(newpoints[i])].append(self.points[i])
        return ps

    def getidx(self, point):
        (x,y,z)=point
        temp=0
        if x>0:
            temp=temp+4
        if y>0:
            temp=temp+2
        if z>0:
            temp=temp+1
        return temp

    def getcornerforpoint(self, point, selfmatrix=None, othermatrix=None):
        if self.singleton:
            return None
        np=self.transform(point,selfmatrix, othermatrix)
        return self.getidx(np)

    def getcorner(self, other, selfmatrix=None, othermatrix=None):
        if self.obb==None:
            return None
        if not other.singleton:
            return self.getcornerforpoint(other.originalcener,selfmatrix,othermatrix)
        return self.getcornerforpoint(other.points[0],selfmatrix,othermatrix)

    def transform(self, point, selfmatrix=None, othermatrix=None):
        if othermatrix!=None:
            point=transferpoint(point, othermatrix)
        newcenter=transferpoint(self.originalcenter,selfmatrix)
        point-=newcenter
        coords=self.obb.rotation
        if othermatrix!=None:
            coords=transfercoord(self.obb.rotation,selfmatrix)
        return np.dot(point, coords)
            
class Object:
    def __init__(self,points,spv_thres):
        self.october=October(points,spv_thres)
        self.tmatrix=None
    
    def update(self,newmatrix):
        if self.tmatrix==None:
            self.tmatrix=newmatrix
        else:
            self.tmatrix=np.dot(newmatrix,self.tmatrix)
    
    @staticmethod
    def cd(Obj1, Obj2):
        o1=Obj1.october
        o2=Obj2.october
        m1=Obj1.tmatrix
        m2=Obj2.tmatrix
        t1=True
        t2=True
        while (t1 or t2):
            if t1:
                d1=o1.getcorner(o2,m1,m2)
                if d1==None or o1.children[d1]==None:
                    t1=False
                else:
                    o1=o1.children[d1]
            if t2:
                d2=o2.getcorner(o1,m2,m1)
                if d2==None or o2.children[d2]==None:
                    t2=False
                else:
                    o2=o2.children[d2]
        if o1.singleton and o2.singleton:
            dist1=transferpoint(o1.upper_cent,m1)-transferpoint(o2.points[0],m2)
            dist1=dist1[0]**2+dist1[1]**2+dist1[2]**2
            limit1=transferpoint(o1.upper_cent,m1)-transferpoint(o1.points[0],m1)
            limit1=limit1[0]**2+limit1[1]**2+limit1[2]**2
            dist2=transferpoint(o2.upper_cent,m2)-transferpoint(o1.points[0],m1)
            dist2=dist2[0]**2+dist2[1]**2+dist2[2]**2
            limit2=transferpoint(o2.upper_cent,m2)-transferpoint(o2.points[0],m2)
            limit2=limit2[0]**2+limit2[1]**2+limit2[2]**2
            if dist1<=limit1 or dist2<=limit2:
                return True
            return False

        elif o1.singleton:
            project=o2.transform(o1.points[0],m2,m1)
            for i in range(3):
                if project[i] < o2.obb.min[i]-o2.obb.centroid[i] or project[i] > o2.obb.max[i]-o2.obb.centroid[i]:
                    return False
            return True
        elif o2.singleton:
            project=o1.transform(o2.points[0],m1,m2)
            for i in range(3):
                if project[i] < o1.obb.min[i]-o2.obb.centroid[i] or project[i] > o1.obb.max[i]-o2.obb.centroid[i]:
                    return False
            return True
        else:
            # This is an approximation instead of getting the real result.
            # To restrict the checking result, use the maximum value as the boundary.
            dist=list(map(abs, o1.originalcenter-o2.originalcenter))
            limit=(o1.obb.max-o1.obb.min+o2.obb.max-o2.obb.min)/2
            for i in range(3):
                if dist[i]>max(limit):
                    return False
            return True

def transferpoint(point, matrix):
    if matrix==None:
        return point
    newpoint=np.array(list(point).append(0))
    newpoint=np.dot(matrix, newpoint)
    return newpoint[:3]

def transfercoord(coord, matrix):
    if matrix==None:
        return coord
    newcoord=[]
    for c in coord:
        newcoord.append(transferpoint(c,matrix))
    return np.array(newcoord)
    