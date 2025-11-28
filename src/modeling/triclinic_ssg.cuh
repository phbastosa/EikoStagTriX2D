# ifndef TRICLINIC_SSG_CUH
# define TRICLINIC_SSG_CUH

# include "triclinic.cuh"

class Triclinic_SSG : public Triclinic
{    
    void set_rec_weights();
    void set_src_weights();

    void set_modeling_type();

    void compute_velocity();
    void compute_pressure();
};

__global__ void float_compute_velocity_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * T, float * B, 
                                           float * damp1D, float * damp2D, float * wavelet, float dx, float dz, float dt, int tId, 
                                           int tlag, int sIdx, int sIdz, float * skw, int nxx, int nzz, int nb, int nt, bool eikonal);

__global__ void float_compute_pressure_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * P, float * T, 
                                           float * C11, float * C13, float * C15, float * C33, float * C35, float * C55, int tId, 
                                           int tlag, float dx, float dz, float dt, int nxx, int nzz, bool eikonal);

__global__ void uintc_compute_velocity_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * T, uintc * B, 
                                           float maxB, float minB, float * damp1D, float * damp2D, float * wavelet, float dx, 
                                           float dz, float dt, int tId, int tlag, int sIdx, int sIdz, float * skw, int nxx, 
                                           int nzz, int nb, int nt, bool eikonal);

__global__ void uintc_compute_pressure_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * P, float * T, 
                                           uintc * C11, uintc * C13, uintc * C15, uintc * C33, uintc * C35, uintc * C55, int tId, 
                                           int tlag, float dx, float dz, float dt, int nxx, int nzz, float minC11, float maxC11, 
                                           float minC13, float maxC13, float minC15, float maxC15, float minC33, float maxC33, 
                                           float minC35, float maxC35, float minC55, float maxC55, bool eikonal);

# endif