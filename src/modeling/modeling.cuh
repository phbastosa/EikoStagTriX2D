# ifndef MODELING_CUH
# define MODELING_CUH

# include <cuda_runtime.h>

# include "../geometry/geometry.hpp"

# define DGS 5

# define NSWEEPS 4
# define MESHDIM 2

# define NTHREADS 256

# define COMPRESS 65535

# define FDM1 6.97545e-4f 
# define FDM2 9.57031e-3f 
# define FDM3 7.97526e-2f 
# define FDM4 1.19628906f 

typedef unsigned short int uintc; 

class Modeling
{
private:

    bool snapshot;
    int snapCount;
    
    std::vector<int> snapId;
    
    float * snapshot_in = nullptr;
    float * snapshot_out = nullptr;

    float * h_seismogram_Ps = nullptr;
    float * h_seismogram_Vx = nullptr;
    float * h_seismogram_Vz = nullptr;

    float * d_seismogram_Ps = nullptr;
    float * d_seismogram_Vx = nullptr;
    float * d_seismogram_Vz = nullptr;

    void set_wavelet();
    void set_dampers();
    void set_eikonal();

    void set_geometry();
    void set_snapshots();
    void set_seismogram();

    void set_wavefields();

    void compute_snapshots();
    void compute_seismogram();

    void show_time_progress();

protected:

    float dx, dz, dt;
    int nxx, nzz, matsize;
    int nt, nx, nz, nb, nPoints;
    int tlag, recId, sIdx, sIdz;
    int nsnap, isnap, fsnap;
    int max_spread, timeId;
    int sBlocks, nBlocks;

    float bd, fmax;

    int total_levels;    

    float sx, sz;
    float dx2i, dz2i;

    int * d_sgnv = nullptr;
    int * d_sgnt = nullptr;

    float * d_skw = nullptr;
    float * d_rkwPs = nullptr;
    float * d_rkwVx = nullptr;
    float * d_rkwVz = nullptr;

    std::string snapshot_folder;
    std::string seismogram_folder;

    std::string modeling_type;
    std::string modeling_name;

    float * S = nullptr;

    float * d1D = nullptr;
    float * d2D = nullptr;

    float * d_S = nullptr;

    float * d_T = nullptr;
    float * d_P = nullptr;
    
    float * d_Vx = nullptr;
    float * d_Vz = nullptr;

    float * d_Txx = nullptr;
    float * d_Tzz = nullptr;
    float * d_Txz = nullptr;

    int * d_rIdx = nullptr;
    int * d_rIdz = nullptr;

    float * d_wavelet = nullptr;

    void eikonal_solver();
    
    virtual void initialization() = 0;
    virtual void compute_eikonal() = 0;
    virtual void compute_velocity() = 0;
    virtual void compute_pressure() = 0;
    virtual void set_specifications() = 0;

    void expand_boundary(float * input, float * output);
    void reduce_boundary(float * input, float * output);
    
    void compression(float * input, uintc * output, int N, float &max_value, float &min_value);

public:

    int srcId;

    std::string parameters;

    Geometry * geometry;

    void set_parameters();
    void show_information();
    void time_propagation();    
    void export_seismogram();
};

__global__ void time_set(float * T, int matsize);

__global__ void time_init(float * T, float * S, float sx, float sz, float dx, 
                          float dz, int sIdx, int sIdz, int nzz, int nb);

__global__ void inner_sweep(float * T, float * S, int * sgnv, int * sgnt, int sgni, int sgnj, 
                            int x_offset, int z_offset, int xd, int zd, int nxx, int nzz, 
                            float dx, float dz, float dx2i, float dz2i);

__global__ void compute_seismogram_GPU(float * P, int * rIdx, int * rIdz, float * rkw, float * seismogram, int spread, int tId, int tlag, int nt, int nzz);

__device__ float get_boundary_damper(float * damp1D, float * damp2D, int i, int j, int nxx, int nzz, int nabc);

# endif