# include "triclinic_ssg.cuh"

void Triclinic_SSG::set_modeling_type()
{
    modeling_name = "Triclinic media with Standard Staggered Grid";
    modeling_type = "triclinic_ssg";
}

void Triclinic_SSG::set_geometry_weights()
{
    for (srcId = 0; srcId < geometry->nsrc; srcId++)
    {
        sx = geometry->xsrc[srcId];
        sz = geometry->zsrc[srcId];

        sIdx = (int)((sx + 0.5f*dx) / dx);
        sIdz = (int)((sz + 0.5f*dz) / dz);

        auto skw = kaiser_weights(sx, sz, sIdx, sIdz, dx, dz);

        for (int zId = 0; zId < DGS; zId++)
            for (int xId = 0; xId < DGS; xId++)
                h_skw[zId + xId*DGS + srcId*DGS*DGS] = skw[zId][xId];
    }
    
    for (recId = 0; recId < geometry->nrec; recId++)
    {
        rx = geometry->xrec[recId];
        rz = geometry->zrec[recId];
        
        rIdx = (int)((rx + 0.5f*dx) / dx);
        rIdz = (int)((rz + 0.5f*dz) / dz);
        
        auto rkwPs = kaiser_weights(rx, rz, rIdx, rIdz, dx, dz);
        auto rkwVx = kaiser_weights(rx + 0.5f*dx, rz, rIdx, rIdz, dx, dz);
        auto rkwVz = kaiser_weights(rx, rz + 0.5f*dz, rIdx, rIdz, dx, dz);
        
        for (int zId = 0; zId < DGS; zId++)
        {
            for (int xId = 0; xId < DGS; xId++)
            {
                h_rkwPs[zId + xId*DGS + recId*DGS*DGS] = rkwPs[zId][xId];
                h_rkwVx[zId + xId*DGS + recId*DGS*DGS] = rkwVx[zId][xId];
                h_rkwVz[zId + xId*DGS + recId*DGS*DGS] = rkwVz[zId][xId];
            }
        }
    }    
}

void Triclinic_SSG::compute_velocity()
{
    if (compression)
    {
        uintc_compute_velocity_ssg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vz,d_Txx,d_Tzz,d_Txz,d_T,dc_B,maxB,minB,d1D,d2D,d_wavelet,
                                                         dx,dz,dt,timeId,tlag,sIdx,sIdz,d_skw,nxx,nzz,nb,nt,eikonalClip);
    }
    else 
    {
        float_compute_velocity_ssg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vz,d_Txx,d_Tzz,d_Txz,d_T,d_B,d1D,d2D,d_wavelet,
                                                         dx,dz,dt,timeId,tlag,sIdx,sIdz,d_skw,nxx,nzz,nb,nt,eikonalClip);
    }
}

void Triclinic_SSG::compute_pressure()
{
    if (compression)
    {
        uintc_compute_pressure_ssg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vz,d_Txx,d_Tzz,d_Txz,d_P,d_T,dc_C11,dc_C13,dc_C15,dc_C33,
                                                         dc_C35,dc_C55,timeId,tlag,dx,dz,dt,nxx,nzz,minC11,maxC11,minC13,
                                                         maxC13,minC15,maxC15,minC33,maxC33,minC35,maxC35,minC55,maxC55, eikonalClip);
    }
    else 
    {
        float_compute_pressure_ssg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vz,d_Txx,d_Tzz,d_Txz,d_P,d_T,d_C11,d_C13,d_C15,d_C33,d_C35,
                                                         d_C55,timeId,tlag,dx,dz,dt,nxx,nzz, eikonalClip);
    }
}

__global__ void uintc_compute_velocity_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * T, uintc * B, float maxB, 
                                           float minB, float * damp1D, float * damp2D, float * wavelet, float dx, float dz, float dt, int tId, 
                                           int tlag, int sIdx, int sIdz, float * skw, int nxx, int nzz, int nb, int nt, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    float Bn, Bm;

    if ((index == 0) && (tId < nt))
    {
        for (int i = 0; i < DGS; i++)
        {
            int zi = sIdz + i - 3;
            for (int j = 0; j < DGS; j++)
            {
                int xi = sIdx + j - 3;

                Txx[zi + xi*nzz] += skw[i + j*DGS]*wavelet[tId] / (dx*dz);
                Tzz[zi + xi*nzz] += skw[i + j*DGS]*wavelet[tId] / (dx*dz);
            }
        }
    }
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {
        Bn = (minB + (static_cast<float>(B[index]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

        if((i >= 3) && (i < nzz-4) && (j > 3) && (j < nxx-3)) 
        {
            float dTxx_dx = (FDM1*(Txx[i + (j-4)*nzz] - Txx[i + (j+3)*nzz]) +
                             FDM2*(Txx[i + (j+2)*nzz] - Txx[i + (j-3)*nzz]) +
                             FDM3*(Txx[i + (j-2)*nzz] - Txx[i + (j+1)*nzz]) +
                             FDM4*(Txx[i + j*nzz]     - Txx[i + (j-1)*nzz])) / dx;

            float dTxz_dz = (FDM1*(Txz[(i-3) + j*nzz] - Txz[(i+4) + j*nzz]) +
                             FDM2*(Txz[(i+3) + j*nzz] - Txz[(i-2) + j*nzz]) +
                             FDM3*(Txz[(i-1) + j*nzz] - Txz[(i+2) + j*nzz]) +
                             FDM4*(Txz[(i+1) + j*nzz] - Txz[i + j*nzz])) / dz;

            Bm = (minB + (static_cast<float>(B[i + (j+1)*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

            float Bx = 0.5f*(Bn + Bm);

            Vx[index] += dt*Bx*(dTxx_dx + dTxz_dz); 
        }

        if((i > 3) && (i < nzz-3) && (j >= 3) && (j < nxx-4)) 
        {
            float dTxz_dx = (FDM1*(Txz[i + (j-3)*nzz] - Txz[i + (j+4)*nzz]) +
                             FDM2*(Txz[i + (j+3)*nzz] - Txz[i + (j-2)*nzz]) +
                             FDM3*(Txz[i + (j-1)*nzz] - Txz[i + (j+2)*nzz]) +
                             FDM4*(Txz[i + (j+1)*nzz] - Txz[i + j*nzz])) / dx;

            float dTzz_dz = (FDM1*(Tzz[(i-4) + j*nzz] - Tzz[(i+3) + j*nzz]) +
                             FDM2*(Tzz[(i+2) + j*nzz] - Tzz[(i-3) + j*nzz]) +
                             FDM3*(Tzz[(i-2) + j*nzz] - Tzz[(i+1) + j*nzz]) +
                             FDM4*(Tzz[i + j*nzz]     - Tzz[(i-1) + j*nzz])) / dz;

            Bm = (minB + (static_cast<float>(B[(i+1) + j*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

            float Bz = 0.5f*(Bn + Bm);

            Vz[index] += dt*Bz*(dTxz_dx + dTzz_dz); 
        }

        float damper = get_boundary_damper(damp1D, damp2D, i, j, nxx, nzz, nb);

        Vx[index] *= damper;
        Vz[index] *= damper;

        Txx[index] *= damper;
        Tzz[index] *= damper;
        Txz[index] *= damper;
    }
}

__global__ void float_compute_velocity_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * T, float * B, 
                                           float * damp1D, float * damp2D, float * wavelet, float dx, float dz, float dt, int tId, 
                                           int tlag, int sIdx, int sIdz, float * skw, int nxx, int nzz, int nb, int nt, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    if ((index == 0) && (tId < nt))
    {
        for (int i = 0; i < DGS; i++)
        {
            int zi = sIdz + i - 3;
            for (int j = 0; j < DGS; j++)
            {
                int xi = sIdx + j - 3;

                Txx[zi + xi*nzz] += skw[i + j*DGS]*wavelet[tId] / (dx*dz);
                Tzz[zi + xi*nzz] += skw[i + j*DGS]*wavelet[tId] / (dx*dz);
            }
        }
    }
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {                
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4))
        {
            float dTxx_dx = (FDM1*(Txx[i + (j-4)*nzz] - Txx[i + (j+3)*nzz]) +
                             FDM2*(Txx[i + (j+2)*nzz] - Txx[i + (j-3)*nzz]) +
                             FDM3*(Txx[i + (j-2)*nzz] - Txx[i + (j+1)*nzz]) +
                             FDM4*(Txx[i + j*nzz]     - Txx[i + (j-1)*nzz])) / dx;

            float dTxz_dz = (FDM1*(Txz[(i-3) + j*nzz] - Txz[(i+4) + j*nzz]) +
                             FDM2*(Txz[(i+3) + j*nzz] - Txz[(i-2) + j*nzz]) +
                             FDM3*(Txz[(i-1) + j*nzz] - Txz[(i+2) + j*nzz]) +
                             FDM4*(Txz[(i+1) + j*nzz] - Txz[i + j*nzz])) / dz;

            float Bx = 0.5f*(B[i + j*nzz] + B[i + (j+1)*nzz]);

            Vx[index] += dt*Bx*(dTxx_dx + dTxz_dz); 
        }

        if((i > 3) && (i < nzz-3) && (j >= 3) && (j < nxx-4)) 
        {
            float dTxz_dx = (FDM1*(Txz[i + (j-3)*nzz] - Txz[i + (j+4)*nzz]) +
                             FDM2*(Txz[i + (j+3)*nzz] - Txz[i + (j-2)*nzz]) +
                             FDM3*(Txz[i + (j-1)*nzz] - Txz[i + (j+2)*nzz]) +
                             FDM4*(Txz[i + (j+1)*nzz] - Txz[i + j*nzz])) / dx;

            float dTzz_dz = (FDM1*(Tzz[(i-4) + j*nzz] - Tzz[(i+3) + j*nzz]) +
                             FDM2*(Tzz[(i+2) + j*nzz] - Tzz[(i-3) + j*nzz]) +
                             FDM3*(Tzz[(i-2) + j*nzz] - Tzz[(i+1) + j*nzz]) +
                             FDM4*(Tzz[i + j*nzz]     - Tzz[(i-1) + j*nzz])) / dz;

            float Bz = 0.5f*(B[i + j*nzz] + B[(i+1) + j*nzz]);

            Vz[index] += dt*Bz*(dTxz_dx + dTzz_dz); 
        }

        float damper = get_boundary_damper(damp1D, damp2D, i, j, nxx, nzz, nb);

        Vx[index] *= damper;
        Vz[index] *= damper;

        Txx[index] *= damper;
        Tzz[index] *= damper;
        Txz[index] *= damper;
    }
}

__global__ void uintc_compute_pressure_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * P, float * T, 
                                           uintc * C11, uintc * C13, uintc * C15, uintc * C33, uintc * C35, uintc * C55, int tId, 
                                           int tlag, float dx, float dz, float dt, int nxx, int nzz, float minC11, float maxC11, 
                                           float minC13, float maxC13, float minC15, float maxC15, float minC33, float maxC33, 
                                           float minC35, float maxC35, float minC55, float maxC55, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    float c15_1, c15_2, c15_3, c15_4;
    float c35_1, c35_2, c35_3, c35_4;
    float c55_1, c55_2, c55_3, c55_4;
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {                
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4))
        {
            float dVx_dx = (FDM1*(Vx[i + (j-3)*nzz] - Vx[i + (j+4)*nzz]) +
                            FDM2*(Vx[i + (j+3)*nzz] - Vx[i + (j-2)*nzz]) +
                            FDM3*(Vx[i + (j-1)*nzz] - Vx[i + (j+2)*nzz]) +
                            FDM4*(Vx[i + (j+1)*nzz] - Vx[i + j*nzz])) / dx;

            float dVx_dz = (FDM1*(Vx[(i-3) + j*nzz] - Vx[(i+4) + j*nzz]) +
                            FDM2*(Vx[(i+3) + j*nzz] - Vx[(i-2) + j*nzz]) +
                            FDM3*(Vx[(i-1) + j*nzz] - Vx[(i+2) + j*nzz]) +
                            FDM4*(Vx[(i+1) + j*nzz] - Vx[i + j*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-3)*nzz] - Vz[i + (j+4)*nzz]) +
                            FDM2*(Vz[i + (j+3)*nzz] - Vz[i + (j-2)*nzz]) +
                            FDM3*(Vz[i + (j-1)*nzz] - Vz[i + (j+2)*nzz]) +
                            FDM4*(Vz[i + (j+1)*nzz] - Vz[i + j*nzz])) / dx;

            float dVz_dz = (FDM1*(Vz[(i-3) + j*nzz] - Vz[(i+4) + j*nzz]) +
                            FDM2*(Vz[(i+3) + j*nzz] - Vz[(i-2) + j*nzz]) +
                            FDM3*(Vz[(i-1) + j*nzz] - Vz[(i+2) + j*nzz]) +
                            FDM4*(Vz[(i+1) + j*nzz] - Vz[i + j*nzz])) / dz;
            
            float c11 = (minC11 + (static_cast<float>(C11[index]) - 1.0f) * (maxC11 - minC11) / (COMPRESS - 1));
            float c13 = (minC13 + (static_cast<float>(C13[index]) - 1.0f) * (maxC13 - minC13) / (COMPRESS - 1));
            float c15 = (minC15 + (static_cast<float>(C15[index]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));
            float c33 = (minC33 + (static_cast<float>(C33[index]) - 1.0f) * (maxC33 - minC33) / (COMPRESS - 1));
            float c35 = (minC35 + (static_cast<float>(C35[index]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));

            Txx[index] += dt*(c11*dVx_dx + c13*dVz_dz + c15*(dVx_dz + dVz_dx));
            Tzz[index] += dt*(c13*dVx_dx + c33*dVz_dz + c35*(dVx_dz + dVz_dx));

            P[index] = 0.5f*(Txx[index] + Tzz[index]);
        }

        if((i > 3) && (i < nzz-3) && (j > 3) && (j < nxx-3)) 
        {
            float dVx_dx = (FDM1*(Vx[i + (j-4)*nzz] - Vx[i + (j+3)*nzz]) +
                            FDM2*(Vx[i + (j+2)*nzz] - Vx[i + (j-3)*nzz]) +
                            FDM3*(Vx[i + (j-2)*nzz] - Vx[i + (j+1)*nzz]) +
                            FDM4*(Vx[i + j*nzz]     - Vx[i + (j-1)*nzz])) / dx;

            float dVx_dz = (FDM1*(Vx[(i-4) + j*nzz] - Vx[(i+3) + j*nzz]) +
                            FDM2*(Vx[(i+2) + j*nzz] - Vx[(i-3) + j*nzz]) +
                            FDM3*(Vx[(i-2) + j*nzz] - Vx[(i+1) + j*nzz]) +
                            FDM4*(Vx[i + j*nzz]     - Vx[(i-1) + j*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-4)*nzz] - Vz[i + (j+3)*nzz]) +
                            FDM2*(Vz[i + (j+2)*nzz] - Vz[i + (j-3)*nzz]) +
                            FDM3*(Vz[i + (j-2)*nzz] - Vz[i + (j+1)*nzz]) +
                            FDM4*(Vz[i + j*nzz]     - Vz[i + (j-1)*nzz])) / dx;

            float dVz_dz = (FDM1*(Vz[(i-4) + j*nzz] - Vz[(i+3) + j*nzz]) +
                            FDM2*(Vz[(i+2) + j*nzz] - Vz[(i-3) + j*nzz]) +
                            FDM3*(Vz[(i-2) + j*nzz] - Vz[(i+1) + j*nzz]) +
                            FDM4*(Vz[i + j*nzz]     - Vz[(i-1) + j*nzz])) / dx;

            c15_1 = (minC15 + (static_cast<float>(C15[(i+1) + (j+1)*nzz]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));
            c15_2 = (minC15 + (static_cast<float>(C15[i + (j+1)*nzz]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));
            c15_3 = (minC15 + (static_cast<float>(C15[(i+1) + j*nzz]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));
            c15_4 = (minC15 + (static_cast<float>(C15[i + j*nzz]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));

            c35_1 = (minC35 + (static_cast<float>(C35[(i+1) + (j+1)*nzz]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));
            c35_2 = (minC35 + (static_cast<float>(C35[i + (j+1)*nzz]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));
            c35_3 = (minC35 + (static_cast<float>(C35[(i+1) + j*nzz]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));
            c35_4 = (minC35 + (static_cast<float>(C35[i + j*nzz]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));

            c55_1 = (minC55 + (static_cast<float>(C55[(i+1) + (j+1)*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_2 = (minC55 + (static_cast<float>(C55[i + (j+1)*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_3 = (minC55 + (static_cast<float>(C55[(i+1) + j*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c55_4 = (minC55 + (static_cast<float>(C55[i + j*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));

            float c15 = 1.0f / (0.25f*(1.0f/c15_1 + 1.0f/c15_2 + 1.0f/c15_3 + 1.0f/c15_4));
            float c35 = 1.0f / (0.25f*(1.0f/c35_1 + 1.0f/c35_2 + 1.0f/c35_3 + 1.0f/c35_4));
            float c55 = 1.0f / (0.25f*(1.0f/c55_1 + 1.0f/c55_2 + 1.0f/c55_3 + 1.0f/c55_4));

            Txz[index] += dt*(c15*dVx_dx + c35*dVz_dz + c55*(dVx_dz + dVz_dx));
        }
    }
}

__global__ void float_compute_pressure_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * P, float * T, 
                                           float * C11, float * C13, float * C15, float * C33, float * C35, float * C55, int tId, 
                                           int tlag, float dx, float dz, float dt, int nxx, int nzz, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {                
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4))
        {
            float dVx_dx = (FDM1*(Vx[i + (j-3)*nzz] - Vx[i + (j+4)*nzz]) +
                            FDM2*(Vx[i + (j+3)*nzz] - Vx[i + (j-2)*nzz]) +
                            FDM3*(Vx[i + (j-1)*nzz] - Vx[i + (j+2)*nzz]) +
                            FDM4*(Vx[i + (j+1)*nzz] - Vx[i + j*nzz])) / dx;

            float dVx_dz = (FDM1*(Vx[(i-3) + j*nzz] - Vx[(i+4) + j*nzz]) +
                            FDM2*(Vx[(i+3) + j*nzz] - Vx[(i-2) + j*nzz]) +
                            FDM3*(Vx[(i-1) + j*nzz] - Vx[(i+2) + j*nzz]) +
                            FDM4*(Vx[(i+1) + j*nzz] - Vx[i + j*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-3)*nzz] - Vz[i + (j+4)*nzz]) +
                            FDM2*(Vz[i + (j+3)*nzz] - Vz[i + (j-2)*nzz]) +
                            FDM3*(Vz[i + (j-1)*nzz] - Vz[i + (j+2)*nzz]) +
                            FDM4*(Vz[i + (j+1)*nzz] - Vz[i + j*nzz])) / dx;

            float dVz_dz = (FDM1*(Vz[(i-3) + j*nzz] - Vz[(i+4) + j*nzz]) +
                            FDM2*(Vz[(i+3) + j*nzz] - Vz[(i-2) + j*nzz]) +
                            FDM3*(Vz[(i-1) + j*nzz] - Vz[(i+2) + j*nzz]) +
                            FDM4*(Vz[(i+1) + j*nzz] - Vz[i + j*nzz])) / dz;
            
            Txx[index] += dt*(C11[index]*dVx_dx + C13[index]*dVz_dz + C15[index]*(dVx_dz + dVz_dx));
            Tzz[index] += dt*(C13[index]*dVx_dx + C33[index]*dVz_dz + C35[index]*(dVx_dz + dVz_dx));
    
            P[index] = 0.5f*(Txx[index] + Tzz[index]);    
        }

        if((i > 3) && (i < nzz-3) && (j > 3) && (j < nxx-3)) 
        {
            float dVx_dx = (FDM1*(Vx[i + (j-4)*nzz] - Vx[i + (j+3)*nzz]) +
                            FDM2*(Vx[i + (j+2)*nzz] - Vx[i + (j-3)*nzz]) +
                            FDM3*(Vx[i + (j-2)*nzz] - Vx[i + (j+1)*nzz]) +
                            FDM4*(Vx[i + j*nzz]     - Vx[i + (j-1)*nzz])) / dx;

            float dVx_dz = (FDM1*(Vx[(i-4) + j*nzz] - Vx[(i+3) + j*nzz]) +
                            FDM2*(Vx[(i+2) + j*nzz] - Vx[(i-3) + j*nzz]) +
                            FDM3*(Vx[(i-2) + j*nzz] - Vx[(i+1) + j*nzz]) +
                            FDM4*(Vx[i + j*nzz]     - Vx[(i-1) + j*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-4)*nzz] - Vz[i + (j+3)*nzz]) +
                            FDM2*(Vz[i + (j+2)*nzz] - Vz[i + (j-3)*nzz]) +
                            FDM3*(Vz[i + (j-2)*nzz] - Vz[i + (j+1)*nzz]) +
                            FDM4*(Vz[i + j*nzz]     - Vz[i + (j-1)*nzz])) / dx;

            float dVz_dz = (FDM1*(Vz[(i-4) + j*nzz] - Vz[(i+3) + j*nzz]) +
                            FDM2*(Vz[(i+2) + j*nzz] - Vz[(i-3) + j*nzz]) +
                            FDM3*(Vz[(i-2) + j*nzz] - Vz[(i+1) + j*nzz]) +
                            FDM4*(Vz[i + j*nzz]     - Vz[(i-1) + j*nzz])) / dx;

            float c15_1 = C15[(i+1) + (j+1)*nzz];
            float c15_2 = C15[i + (j+1)*nzz];
            float c15_3 = C15[(i+1) + j*nzz];
            float c15_4 = C15[i + j*nzz];

            float c35_1 = C35[(i+1) + (j+1)*nzz];
            float c35_2 = C35[i + (j+1)*nzz];
            float c35_3 = C35[(i+1) + j*nzz];
            float c35_4 = C35[i + j*nzz];

            float c55_1 = C55[(i+1) + (j+1)*nzz];
            float c55_2 = C55[i + (j+1)*nzz];
            float c55_3 = C55[(i+1) + j*nzz];
            float c55_4 = C55[i + j*nzz];

            float c15 = 1.0f / (0.25f*(1.0f/c15_1 + 1.0f/c15_2 + 1.0f/c15_3 + 1.0f/c15_4));
            float c35 = 1.0f / (0.25f*(1.0f/c35_1 + 1.0f/c35_2 + 1.0f/c35_3 + 1.0f/c35_4));
            float c55 = 1.0f / (0.25f*(1.0f/c55_1 + 1.0f/c55_2 + 1.0f/c55_3 + 1.0f/c55_4));

            Txz[index] += dt*(c15*dVx_dx + c35*dVz_dz + c55*(dVx_dz + dVz_dx));
        }
    }
}
