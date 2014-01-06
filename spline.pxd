cdef class Spline3D:
    cdef:
        double[:] x, y, z
        double[:,:,:] k, f
        unsigned int nx, ny, nz, np
    cdef:
        void make(self, object x, object y, object z, double[:,:,:] f)
        void load(self, folderName)
        void save(self, folderName)
        double interpolate(self, double x, double y, double z)