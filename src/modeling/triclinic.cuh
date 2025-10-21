# ifndef TRICLINIC_CUH
# define TRICLINIC_CUH

# include <cuda_runtime.h>

# include "../geometry/geometry.hpp"

# define DGS 8

# define NSWEEPS 4
# define MESHDIM 2

# define NTHREADS 256

# define COMPRESS 65535

# define FDM1 6.97545e-4f 
# define FDM2 9.57031e-3f 
# define FDM3 7.97526e-2f 
# define FDM4 1.19628906f 

typedef unsigned short int uintc; 

class Triclinic
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

    void set_models();
    void set_wavelet();
    void set_dampers();
    void set_eikonal();
    void set_slowness();

    void set_geometry();
    void set_snapshots();
    void set_seismogram();
    void set_wavefields();

    void eikonal_solver();
    void time_propagation();
    void wavefield_refresh();
    void get_shot_position();
    
    void compute_eikonal();
    void compute_snapshots();
    void compute_seismogram();
    void export_seismograms();
    void export_travelTimes();

    void show_information();
    void show_time_progress();

    void expand_boundary(float * input, float * output);
    void reduce_boundary(float * input, float * output);
    
    void get_compression(float * input, uintc * output, int N, float &max_value, float &min_value);

protected:

    float dx, dz, dt;
    int nxx, nzz, matsize;
    int nt, nx, nz, nb, nPoints;
    int sIdx, sIdz, srcId, recId;
    int tlag, nsnap, isnap, fsnap;
    int max_spread, timeId;
    int sBlocks, nBlocks;
    int total_levels;    

    float sx, sz;
    float bd, fmax; 
    float dx2i, dz2i;

    bool eikonalClip; 
    bool compression;

    int * d_sgnv = nullptr;
    int * d_sgnt = nullptr;

    float * d_skw = nullptr;
    float * d_rkwPs = nullptr;
    float * d_rkwVx = nullptr;
    float * d_rkwVz = nullptr;

    float * d_wavelet = nullptr;

    int * d_rIdx = nullptr;
    int * d_rIdz = nullptr;

    Geometry * geometry;

    std::string modeling_type;
    std::string modeling_name;

    std::string snapshot_folder;
    std::string seismogram_folder;

    float * d1D = nullptr;
    float * d2D = nullptr;

    float * h_S = nullptr;
    float * d_S = nullptr; 
    float * d_T = nullptr;
    float * d_P = nullptr;
    
    float * d_Vx = nullptr;
    float * d_Vz = nullptr;

    float * d_Txx = nullptr;
    float * d_Tzz = nullptr;
    float * d_Txz = nullptr;

    float * d_B = nullptr; uintc * dc_B = nullptr; float maxB; float minB;
    
    float * d_C11 = nullptr; uintc * dc_C11 = nullptr; float maxC11; float minC11;
    float * d_C13 = nullptr; uintc * dc_C13 = nullptr; float maxC13; float minC13;
    float * d_C15 = nullptr; uintc * dc_C15 = nullptr; float maxC15; float minC15;
    float * d_C33 = nullptr; uintc * dc_C33 = nullptr; float maxC33; float minC33;
    float * d_C35 = nullptr; uintc * dc_C35 = nullptr; float maxC35; float minC35;
    float * d_C55 = nullptr; uintc * dc_C55 = nullptr; float maxC55; float minC55;

    virtual void initialization() = 0;
    virtual void compute_velocity() = 0;
    virtual void compute_pressure() = 0;

public:

    std::string parameters;

    void set_parameters();
    void run_wave_propagation();
};

__global__ void time_set(float * T, int matsize);

__global__ void time_init(float * T, float * S, float sx, float sz, float dx, 
                          float dz, int sIdx, int sIdz, int nzz, int nb);

__global__ void inner_sweep(float * T, float * S, int * sgnv, int * sgnt, int sgni, int sgnj, 
                            int x_offset, int z_offset, int xd, int zd, int nxx, int nzz, 
                            float dx, float dz, float dx2i, float dz2i);

__global__ void float_quasi_slowness(float * T, float * S, float dx, float dz, int sIdx, int sIdz, int nxx, int nzz, 
                                     int nb, float * C11, float * C13, float * C15, float * C33, float * C35, float * C55);

__global__ void uintc_quasi_slowness(float * T, float * S, float dx, float dz, int sIdx, int sIdz, int nxx, int nzz, 
                                     int nb, uintc * C11, uintc * C13, uintc * C15, uintc * C33, uintc * C35, uintc * C55, 
                                     float minC11, float maxC11, float minC13, float maxC13, float minC15, float maxC15, 
                                     float minC33, float maxC33, float minC35, float maxC35, float minC55, float maxC55);

__global__ void compute_seismogram_GPU(float * P, int * rIdx, int * rIdz, float * rkw, float * seismogram, int spread, int tId, int tlag, int nt, int nzz);

__device__ float get_boundary_damper(float * damp1D, float * damp2D, int i, int j, int nxx, int nzz, int nabc);

# endif