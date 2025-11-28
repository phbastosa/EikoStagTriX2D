# include "triclinic_rsg.cuh"

void Triclinic_RSG::set_modeling_type()
{
    modeling_name = "Triclinic media with Rotated Staggered Grid";
    modeling_type = "triclinic_rsg";
}

void Triclinic_RSG::set_rec_weights()
{
    int * h_rIdx = new int[geometry->nrec]();
    int * h_rIdz = new int[geometry->nrec]();

    float * h_rkwPs = new float[DGS*DGS*geometry->nrec]();
    //float * h_rkwVx = new float[DGS*DGS*geometry->nrec]();
    //float * h_rkwVz = new float[DGS*DGS*geometry->nrec]();

    for (recId = 0; recId < geometry->nrec; recId++)
    {
        float rx = geometry->xrec[recId];
        float rz = geometry->zrec[recId];
        
        int rIdx = (int)((rx + 0.5f*dx) / dx);
        int rIdz = (int)((rz + 0.5f*dz) / dz);
    
        auto rkwPs = kaiser_weights(rx, rz, rIdx, rIdz, dx, dz);        
        //auto rkwVx = kaiser_weights(rx + 0.5f*dx, rz + 0.5f*dz, rIdx, rIdz, dx, dz);
        //auto rkwVz = kaiser_weights(rx + 0.5f*dx, rz + 0.5f*dz, rIdx, rIdz, dx, dz);
        
        for (int zId = 0; zId < DGS; zId++)
        {
            for (int xId = 0; xId < DGS; xId++)
            {
                h_rkwPs[zId + xId*DGS + recId*DGS*DGS] = rkwPs[zId][xId];
                //h_rkwVx[zId + xId*DGS + recId*DGS*DGS] = rkwVx[zId][xId];
                //h_rkwVz[zId + xId*DGS + recId*DGS*DGS] = rkwVz[zId][xId];
            }
        }

        h_rIdx[recId] = rIdx + nb;
        h_rIdz[recId] = rIdz + nb;
    }

    cudaMemcpy(d_rkwPs, h_rkwPs, DGS*DGS*geometry->nrec*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_rkwVx, h_rkwVx, DGS*DGS*geometry->nrec*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_rkwVz, h_rkwVz, DGS*DGS*geometry->nrec*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_rIdx, h_rIdx, geometry->nrec*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdz, h_rIdz, geometry->nrec*sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_rkwPs;
    //delete[] h_rkwVx;
    //delete[] h_rkwVz;
    delete[] h_rIdx;
    delete[] h_rIdz;
}

void Triclinic_RSG::set_src_weights()
{
    float * h_skw = new float[DGS*DGS]();

        auto skw = gaussian_weights(sx, sz, sIdx, sIdz, dx, dz);

    for (int xId = 0; xId < DGS; xId++)
        for (int zId = 0; zId < DGS; zId++)
            h_skw[zId + xId*DGS] = skw[zId][xId];

    cudaMemcpy(d_skw, h_skw, DGS*DGS*sizeof(float), cudaMemcpyHostToDevice);
    
    delete[] h_skw;
}

void Triclinic_RSG::compute_velocity()
{
    if (compression)
    {
        uintc_compute_velocity_rsg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vz,d_Txx,d_Tzz,d_Txz,d_T,dc_B,maxB,minB,d1D,d2D,d_wavelet,
                                                         dx,dz,dt,timeId,tlag,sIdx,sIdz,d_skw,nxx,nzz,nb,nt, eikonalClip);
    }
    else 
    {
        float_compute_velocity_rsg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vz,d_Txx,d_Tzz,d_Txz,d_T,d_B,d1D,d2D,d_wavelet,
                                                         dx,dz,dt,timeId,tlag,sIdx,sIdz,d_skw,nxx,nzz,nb,nt, eikonalClip);
    }
}

void Triclinic_RSG::compute_pressure()
{
    if (compression)
    {
        uintc_compute_pressure_rsg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vz,d_Txx,d_Tzz,d_Txz,d_P,d_T,dc_C11,dc_C13,dc_C15,dc_C33,
                                                         dc_C35,dc_C55,timeId,tlag,dx,dz,dt,nxx,nzz,minC11,maxC11,minC13,
                                                         maxC13,minC15,maxC15,minC33,maxC33,minC35,maxC35,minC55,maxC55, eikonalClip);
    }
    else 
    {
        float_compute_pressure_rsg<<<nBlocks,NTHREADS>>>(d_Vx,d_Vz,d_Txx,d_Tzz,d_Txz,d_P,d_T,d_C11,d_C13,d_C15,d_C33,d_C35,
                                                         d_C55,timeId,tlag,dx,dz,dt,nxx,nzz, eikonalClip);
    }
}

__global__ void uintc_compute_velocity_rsg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * T, uintc * B, 
                                           float maxB, float minB, float * damp1D, float * damp2D, float * wavelet, float dx, 
                                           float dz, float dt, int tId, int tlag, int sIdx, int sIdz, float * skw, int nxx, 
                                           int nzz, int nb, int nt, bool eikonal)
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

    float d1_Txx = 0.0f; float d2_Txx = 0.0f;
    float d1_Tzz = 0.0f; float d2_Tzz = 0.0f;
    float d1_Txz = 0.0f; float d2_Txz = 0.0f;
 
    float FDM[] = {FDM4, -FDM3, FDM2, -FDM1};
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {                
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4)) 
        {   
            # pragma unroll 4 
            for (int rsg = 0; rsg < 4; rsg++)
            {
                d1_Txx += FDM[rsg]*(Txx[(i+rsg+1) + (j+rsg+1)*nzz] - Txx[(i-rsg) + (j-rsg)*nzz]);
                d1_Tzz += FDM[rsg]*(Tzz[(i+rsg+1) + (j+rsg+1)*nzz] - Tzz[(i-rsg) + (j-rsg)*nzz]);
                d1_Txz += FDM[rsg]*(Txz[(i+rsg+1) + (j+rsg+1)*nzz] - Txz[(i-rsg) + (j-rsg)*nzz]);

                d2_Txx += FDM[rsg]*(Txx[(i-rsg) + (j+rsg+1)*nzz] - Txx[(i+rsg+1) + (j-rsg)*nzz]);
                d2_Tzz += FDM[rsg]*(Tzz[(i-rsg) + (j+rsg+1)*nzz] - Tzz[(i+rsg+1) + (j-rsg)*nzz]);
                d2_Txz += FDM[rsg]*(Txz[(i-rsg) + (j+rsg+1)*nzz] - Txz[(i+rsg+1) + (j-rsg)*nzz]);
            }
    
            float dTxx_dx = 0.5f*(d1_Txx + d2_Txx) / dx;
            float dTxz_dx = 0.5f*(d1_Txz + d2_Txz) / dx;

            float dTxz_dz = 0.5f*(d1_Txz - d2_Txz) / dz;
            float dTzz_dz = 0.5f*(d1_Tzz - d2_Tzz) / dz;

            float B00 = (minB + (static_cast<float>(B[i + j*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
            float B10 = (minB + (static_cast<float>(B[i + (j+1)*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));;
            float B01 = (minB + (static_cast<float>(B[(i+1) + j*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));;
            float B11 = (minB + (static_cast<float>(B[(i+1) + (j+1)*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));;

            float Bxz = 0.25f*(B00 + B10 + B01 + B11);

            Vx[index] += dt*Bxz*(dTxx_dx + dTxz_dz); 
            Vz[index] += dt*Bxz*(dTxz_dx + dTzz_dz);    
            
            float damper = get_boundary_damper(damp1D, damp2D, i, j, nxx, nzz, nb);

            Vx[index] *= damper;
            Vz[index] *= damper;

            Txx[index] *= damper;
            Tzz[index] *= damper;
            Txz[index] *= damper;
        }
    }
}

__global__ void float_compute_velocity_rsg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * T, float * B, 
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

    float d1_Txx = 0.0f; float d2_Txx = 0.0f;
    float d1_Tzz = 0.0f; float d2_Tzz = 0.0f;
    float d1_Txz = 0.0f; float d2_Txz = 0.0f;
 
    float FDM[] = {FDM4, -FDM3, FDM2, -FDM1};
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {                
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4)) 
        {   
            # pragma unroll 4 
            for (int rsg = 0; rsg < 4; rsg++)
            {
                d1_Txx += FDM[rsg]*(Txx[(i+rsg+1) + (j+rsg+1)*nzz] - Txx[(i-rsg) + (j-rsg)*nzz]);
                d1_Tzz += FDM[rsg]*(Tzz[(i+rsg+1) + (j+rsg+1)*nzz] - Tzz[(i-rsg) + (j-rsg)*nzz]);
                d1_Txz += FDM[rsg]*(Txz[(i+rsg+1) + (j+rsg+1)*nzz] - Txz[(i-rsg) + (j-rsg)*nzz]);

                d2_Txx += FDM[rsg]*(Txx[(i-rsg) + (j+rsg+1)*nzz] - Txx[(i+rsg+1) + (j-rsg)*nzz]);
                d2_Tzz += FDM[rsg]*(Tzz[(i-rsg) + (j+rsg+1)*nzz] - Tzz[(i+rsg+1) + (j-rsg)*nzz]);
                d2_Txz += FDM[rsg]*(Txz[(i-rsg) + (j+rsg+1)*nzz] - Txz[(i+rsg+1) + (j-rsg)*nzz]);
            }
    
            float dTxx_dx = 0.5f*(d1_Txx + d2_Txx) / dx;
            float dTxz_dx = 0.5f*(d1_Txz + d2_Txz) / dx;

            float dTxz_dz = 0.5f*(d1_Txz - d2_Txz) / dz;
            float dTzz_dz = 0.5f*(d1_Tzz - d2_Tzz) / dz;

            float B00 = B[i + j*nzz];
            float B10 = B[i + (j+1)*nzz];
            float B01 = B[(i+1) + j*nzz];
            float B11 = B[(i+1) + (j+1)*nzz];

            float Bxz = 0.25f*(B00 + B10 + B01 + B11);

            Vx[index] += dt*Bxz*(dTxx_dx + dTxz_dz); 
            Vz[index] += dt*Bxz*(dTxz_dx + dTzz_dz);    
            
            float damper = get_boundary_damper(damp1D, damp2D, i, j, nxx, nzz, nb);

            Vx[index] *= damper;
            Vz[index] *= damper;

            Txx[index] *= damper;
            Tzz[index] *= damper;
            Txz[index] *= damper;
        }
    }
}

__global__ void uintc_compute_pressure_rsg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * P, float * T, 
                                           uintc * C11, uintc * C13, uintc * C15, uintc * C33, uintc * C35, uintc * C55, int tId, 
                                           int tlag, float dx, float dz, float dt, int nxx, int nzz, float minC11, float maxC11, 
                                           float minC13, float maxC13, float minC15, float maxC15, float minC33, float maxC33, 
                                           float minC35, float maxC35, float minC55, float maxC55, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    float d1_Vx = 0.0f; float d2_Vx = 0.0f;
    float d1_Vz = 0.0f; float d2_Vz = 0.0f;

    float FDM[] = {FDM4, -FDM3, FDM2, -FDM1};
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {                
        if((i > 3) && (i < nzz-3) && (j > 3) && (j < nxx-3)) 
        {
            # pragma unroll 4
            for (int rsg = 0; rsg < 4; rsg++)
            {       
                d1_Vx += FDM[rsg]*(Vx[(i+rsg) + (j+rsg)*nzz] - Vx[(i-rsg-1) + (j-rsg-1)*nzz]);      
                d1_Vz += FDM[rsg]*(Vz[(i+rsg) + (j+rsg)*nzz] - Vz[(i-rsg-1) + (j-rsg-1)*nzz]);      
    
                d2_Vx += FDM[rsg]*(Vx[(i-rsg-1) + (j+rsg)*nzz] - Vx[(i+rsg) + (j-rsg-1)*nzz]);      
                d2_Vz += FDM[rsg]*(Vz[(i-rsg-1) + (j+rsg)*nzz] - Vz[(i+rsg) + (j-rsg-1)*nzz]);      
            }
    
            float dVx_dx = 0.5f*(d1_Vx + d2_Vx) / dx;
            float dVz_dx = 0.5f*(d1_Vz + d2_Vz) / dx;
            
            float dVx_dz = 0.5f*(d1_Vx - d2_Vx) / dz;
            float dVz_dz = 0.5f*(d1_Vz - d2_Vz) / dz;

            float c11 = (minC11 + (static_cast<float>(C11[index]) - 1.0f) * (maxC11 - minC11) / (COMPRESS - 1));
            float c13 = (minC13 + (static_cast<float>(C13[index]) - 1.0f) * (maxC13 - minC13) / (COMPRESS - 1));
            float c15 = (minC15 + (static_cast<float>(C15[index]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));
            float c33 = (minC33 + (static_cast<float>(C33[index]) - 1.0f) * (maxC33 - minC33) / (COMPRESS - 1));
            float c35 = (minC35 + (static_cast<float>(C35[index]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));    
            float c55 = (minC55 + (static_cast<float>(C55[index]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
                    
            Txx[index] += dt*(c11*dVx_dx + c13*dVz_dz + c15*(dVx_dz + dVz_dx));
            Tzz[index] += dt*(c13*dVx_dx + c33*dVz_dz + c35*(dVx_dz + dVz_dx));
            Txz[index] += dt*(c15*dVx_dx + c35*dVz_dz + c55*(dVx_dz + dVz_dx));
        
            P[index] = 0.5f*(Txx[index] + Tzz[index]);
        }
    }
}

__global__ void float_compute_pressure_rsg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * P, float * T, 
                                           float * C11, float * C13, float * C15, float * C33, float * C35, float * C55, int tId, 
                                           int tlag, float dx, float dz, float dt, int nxx, int nzz, bool eikonal)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    float d1_Vx = 0.0f; float d2_Vx = 0.0f;
    float d1_Vz = 0.0f; float d2_Vz = 0.0f;

    float FDM[] = {FDM4, -FDM3, FDM2, -FDM1};
    
    T[index] = (eikonal) ? T[index] : 0.0f;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {                
        if((i > 3) && (i < nzz-3) && (j > 3) && (j < nxx-3)) 
        {
            # pragma unroll 4
            for (int rsg = 0; rsg < 4; rsg++)
            {       
                d1_Vx += FDM[rsg]*(Vx[(i+rsg) + (j+rsg)*nzz] - Vx[(i-rsg-1) + (j-rsg-1)*nzz]);      
                d1_Vz += FDM[rsg]*(Vz[(i+rsg) + (j+rsg)*nzz] - Vz[(i-rsg-1) + (j-rsg-1)*nzz]);      
    
                d2_Vx += FDM[rsg]*(Vx[(i-rsg-1) + (j+rsg)*nzz] - Vx[(i+rsg) + (j-rsg-1)*nzz]);      
                d2_Vz += FDM[rsg]*(Vz[(i-rsg-1) + (j+rsg)*nzz] - Vz[(i+rsg) + (j-rsg-1)*nzz]);      
            }
    
            float dVx_dx = 0.5f*(d1_Vx + d2_Vx) / dx;
            float dVz_dx = 0.5f*(d1_Vz + d2_Vz) / dx;
            
            float dVx_dz = 0.5f*(d1_Vx - d2_Vx) / dz;
            float dVz_dz = 0.5f*(d1_Vz - d2_Vz) / dz;
                    
            Txx[index] += dt*(C11[index]*dVx_dx + C13[index]*dVz_dz + C15[index]*(dVx_dz + dVz_dx));
            Tzz[index] += dt*(C13[index]*dVx_dx + C33[index]*dVz_dz + C35[index]*(dVx_dz + dVz_dx));
            Txz[index] += dt*(C15[index]*dVx_dx + C35[index]*dVz_dz + C55[index]*(dVx_dz + dVz_dx));
        
            P[index] = 0.5f*(Txx[index] + Tzz[index]);
        }
    }
}
