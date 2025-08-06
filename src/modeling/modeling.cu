# include "modeling.cuh"

void Modeling::set_parameters()
{
    nx = std::stoi(catch_parameter("x_samples", parameters));
    nz = std::stoi(catch_parameter("z_samples", parameters));

    dx = std::stof(catch_parameter("x_spacing", parameters));
    dz = std::stof(catch_parameter("z_spacing", parameters));

    nt = std::stoi(catch_parameter("time_samples", parameters));
    dt = std::stof(catch_parameter("time_spacing", parameters));
    
    fmax = std::stof(catch_parameter("max_frequency", parameters));

    nb = std::stoi(catch_parameter("boundary_samples", parameters));
    bd = std::stof(catch_parameter("boundary_damping", parameters));
    
    isnap = std::stoi(catch_parameter("beg_snap", parameters));
    fsnap = std::stoi(catch_parameter("end_snap", parameters));
    nsnap = std::stoi(catch_parameter("num_snap", parameters));

    snapshot = str2bool(catch_parameter("snapshot", parameters));

    snapshot_folder = catch_parameter("snapshot_folder", parameters);
    seismogram_folder = catch_parameter("seismogram_folder", parameters);

    nPoints = nx*nz;

    nxx = nx + 2*nb;
    nzz = nz + 2*nb;

    matsize = nxx*nzz;

    nBlocks = (int)((matsize + NTHREADS - 1) / NTHREADS);

    set_wavelet();
    set_dampers();
    set_eikonal();

    set_geometry();
    set_snapshots();
    set_seismogram();

    set_specifications();

    cudaMalloc((void**)&(d_P), matsize*sizeof(float));
    cudaMalloc((void**)&(d_T), matsize*sizeof(float));

    cudaMalloc((void**)&(d_Vx), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Vz), matsize*sizeof(float));

    cudaMalloc((void**)&(d_Txx), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Tzz), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Txz), matsize*sizeof(float));
}

void Modeling::set_wavelet()
{
    float * signal_aux1 = new float[nt]();
    float * signal_aux2 = new float[nt]();

    float t0 = 2.0f*sqrtf(M_PI) / fmax;
    float fc = fmax / (3.0f * sqrtf(M_PI));

    tlag = (int)((t0 - 0.5f*dt) / dt) - 1;

    for (int n = 0; n < nt; n++)
    {
        float td = n*dt - t0;

        float arg = M_PI*M_PI*M_PI*fc*fc*td*td;

        signal_aux1[n] = 1e5f*(1.0f - 2.0f*arg)*expf(-arg);
    }

    for (int n = 0; n < nt; n++)
    {
        float summation = 0;
        for (int i = 0; i < n; i++)
            summation += signal_aux1[i];    
        
        signal_aux2[n] = summation;
    }

    double * time_domain = (double *) fftw_malloc(nt*sizeof(double));

    fftw_complex * freq_domain = (fftw_complex *) fftw_malloc(nt*sizeof(fftw_complex));

    fftw_plan forward_plan = fftw_plan_dft_r2c_1d(nt, time_domain, freq_domain, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_c2r_1d(nt, freq_domain, time_domain, FFTW_ESTIMATE);

    double df = 1.0 / (nt * dt);  
    
    std::complex<double> j(0.0, 1.0);  

    for (int k = 0; k < nt; k++) time_domain[k] = (double) signal_aux2[k];

    fftw_execute(forward_plan);

    for (int k = 0; k < nt; ++k) 
    {
        double f = (k <= nt / 2) ? k * df : (k - nt) * df;
        
        std::complex<double> half_derivative_filter = std::pow(2.0 * M_PI * f * j, 0.5);  

        std::complex<double> complex_freq(freq_domain[k][0], freq_domain[k][1]);
        std::complex<double> filtered_freq = complex_freq * half_derivative_filter;

        freq_domain[k][0] = filtered_freq.real();
        freq_domain[k][1] = filtered_freq.imag();
    }

    fftw_execute(inverse_plan);    

    for (int k = 0; k < nt; k++) signal_aux1[k] = (float) time_domain[k] / nt;

    cudaMalloc((void**)&(d_wavelet), nt*sizeof(float));

    cudaMemcpy(d_wavelet, signal_aux1, nt*sizeof(float), cudaMemcpyHostToDevice);

    delete[] signal_aux1;
    delete[] signal_aux2;
}

void Modeling::set_dampers()
{
    float * damp1D = new float[nb]();
    float * damp2D = new float[nb*nb]();

    for (int i = 0; i < nb; i++) 
    {
        damp1D[i] = expf(-powf(bd * (nb - i), 2.0f));
    }

    for(int i = 0; i < nb; i++) 
    {
        for (int j = 0; j < nb; j++)
        {   
            damp2D[j + i*nb] += damp1D[i];
            damp2D[i + j*nb] += damp1D[i];
        }
    }

    for (int index = 0; index < nb*nb; index++)
        damp2D[index] -= 1.0f;

	cudaMalloc((void**)&(d1D), nb*sizeof(float));
	cudaMalloc((void**)&(d2D), nb*nb*sizeof(float));

	cudaMemcpy(d1D, damp1D, nb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d2D, damp2D, nb*nb*sizeof(float), cudaMemcpyHostToDevice);

    delete[] damp1D;
    delete[] damp2D;
}

void Modeling::set_eikonal()
{
    dz2i = 1.0f / (dz * dz);
    dx2i = 1.0f / (dx * dx);

    total_levels = (nxx - 1) + (nzz - 1);

    std::vector<std::vector<int>> sgnv = {{1, 1}, {0, 1},  {1, 0}, {0, 0}};
    std::vector<std::vector<int>> sgnt = {{1, 1}, {-1, 1}, {1, -1}, {-1, -1}};

    int * h_sgnv = new int [NSWEEPS*MESHDIM]();
    int * h_sgnt = new int [NSWEEPS*MESHDIM](); 

    for (int index = 0; index < NSWEEPS*MESHDIM; index++)
    {
        int j = index / NSWEEPS;
    	int i = index % NSWEEPS;				

	    h_sgnv[i + j*NSWEEPS] = sgnv[i][j];
	    h_sgnt[i + j*NSWEEPS] = sgnt[i][j];    
    }

    cudaMalloc((void**)&(d_T), matsize*sizeof(float));
    cudaMalloc((void**)&(d_S), matsize*sizeof(float));

    cudaMalloc((void**)&(d_sgnv), NSWEEPS*MESHDIM*sizeof(int));
    cudaMalloc((void**)&(d_sgnt), NSWEEPS*MESHDIM*sizeof(int));
    
    cudaMemcpy(d_S, S, matsize*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_sgnv, h_sgnv, NSWEEPS*MESHDIM*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sgnt, h_sgnt, NSWEEPS*MESHDIM*sizeof(int), cudaMemcpyHostToDevice);

    std::vector<std::vector<int>>().swap(sgnv);
    std::vector<std::vector<int>>().swap(sgnt);

    delete[] h_sgnt;
    delete[] h_sgnv;
}

void Modeling::set_geometry()
{
    geometry = new Geometry();
    geometry->parameters = parameters;
    geometry->set_parameters();

    max_spread = 0;
    for (int index = 0; index < geometry->nrel; index++)
    {   
        if (max_spread < geometry->spread[index])
            max_spread = geometry->spread[index]; 
    }

    cudaMalloc((void**)&(d_rIdx), max_spread*sizeof(int));
    cudaMalloc((void**)&(d_rIdz), max_spread*sizeof(int));

    cudaMalloc((void**)&(d_skw), DGS*DGS*sizeof(float));
    
    cudaMalloc((void**)&(d_rkwPs), DGS*DGS*max_spread*sizeof(float));
    cudaMalloc((void**)&(d_rkwVx), DGS*DGS*max_spread*sizeof(float));
    cudaMalloc((void**)&(d_rkwVz), DGS*DGS*max_spread*sizeof(float));
}

void Modeling::set_snapshots()
{
    if (snapshot)
    {
        if (nsnap == 1) 
            snapId.push_back(isnap);
        else 
        {
            for (int i = 0; i < nsnap; i++) 
                snapId.push_back(isnap + i * (fsnap - isnap) / (nsnap - 1));
        }
        
        snapshot_in = new float[matsize]();
        snapshot_out = new float[nPoints]();
    }
}

void Modeling::set_seismogram()
{
    sBlocks = (int)((max_spread + NTHREADS - 1) / NTHREADS); 

    h_seismogram_Ps = new float[nt*max_spread]();
    h_seismogram_Vx = new float[nt*max_spread]();
    h_seismogram_Vz = new float[nt*max_spread]();

    cudaMalloc((void**)&(d_seismogram_Ps), nt*max_spread*sizeof(float));
    cudaMalloc((void**)&(d_seismogram_Vx), nt*max_spread*sizeof(float));
    cudaMalloc((void**)&(d_seismogram_Vz), nt*max_spread*sizeof(float));
}

void Modeling::time_propagation()
{
    set_wavefields();
    initialization();
    compute_eikonal();

    if (snapshot)
    {
        snapCount = 0;

        cudaMemcpy(snapshot_in, d_T, matsize*sizeof(float), cudaMemcpyDeviceToHost);
        reduce_boundary(snapshot_in, snapshot_out);
        export_binary_float(snapshot_folder + modeling_type + "_eikonal_" + std::to_string(nz) + "x" + std::to_string(nx) + "_shot_" + std::to_string(geometry->sInd[srcId]+1) + ".bin", snapshot_out, nPoints);    
    }

    for (timeId = 0; timeId < nt + tlag; timeId++)
    {
        compute_velocity();
        compute_pressure();
        compute_snapshots();
        compute_seismogram();    
        show_time_progress();
    }
}

void Modeling::set_wavefields()
{
    cudaMemset(d_P, 0.0f, matsize*sizeof(float));
    
    cudaMemset(d_Vx, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Vz, 0.0f, matsize*sizeof(float));
    
    cudaMemset(d_Txx, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Tzz, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Txz, 0.0f, matsize*sizeof(float));
}

void Modeling::eikonal_solver()
{
    int min_level = std::min(nxx, nzz);
    int max_level = std::max(nxx, nzz);

    int z_offset, x_offset, n_elements;

    for (int sweep = 0; sweep < NSWEEPS; sweep++)
    { 
        int zd = (sweep == 2 || sweep == 3) ? -1 : 1; 
        int xd = (sweep == 0 || sweep == 2) ? -1 : 1;

        int sgni = sweep + 0*NSWEEPS;
        int sgnj = sweep + 1*NSWEEPS;

        for (int level = 0; level < total_levels; level++)
        {
            z_offset = (sweep == 0) ? ((level < nxx) ? 0 : level - nxx + 1) :
                       (sweep == 1) ? ((level < nzz) ? nzz - level - 1 : 0) :
                       (sweep == 2) ? ((level < nzz) ? level : nzz - 1) :
                                      ((level < nxx) ? nzz - 1 : nzz - 1 - (level - nxx + 1));

            x_offset = (sweep == 0) ? ((level < nxx) ? level : nxx - 1) :
                       (sweep == 1) ? ((level < nzz) ? 0 : level - nzz + 1) :
                       (sweep == 2) ? ((level < nzz) ? nxx - 1 : nxx - 1 - (level - nzz + 1)) :
                                      ((level < nxx) ? nxx - level - 1 : 0);

            n_elements = (level < min_level) ? level + 1 : 
                         (level >= max_level) ? total_levels - level : 
                         total_levels - min_level - max_level + level;

            int nblk = (int)((n_elements + NTHREADS - 1) / NTHREADS);

            inner_sweep<<<nblk, NTHREADS>>>(d_T, d_S, d_sgnv, d_sgnt, sgni, sgnj, x_offset, z_offset, xd, zd, nxx, nzz, dx, dz, dx2i, dz2i); 
        }
    }
}

void Modeling::compute_snapshots()
{
    if (snapshot)
    {
        if (snapCount < snapId.size())
        {
            if ((timeId-tlag) == snapId[snapCount])
            {
                cudaMemcpy(snapshot_in, d_P, matsize*sizeof(float), cudaMemcpyDeviceToHost);
                reduce_boundary(snapshot_in, snapshot_out);
                export_binary_float(snapshot_folder + modeling_type + "_snapshot_step" + std::to_string(timeId-tlag) + "_" + std::to_string(nz) + "x" + std::to_string(nx) + "_shot_" + std::to_string(geometry->sInd[srcId]+1) + ".bin", snapshot_out, nPoints);    
                
                ++snapCount;
            }
        }
    }
}

void Modeling::show_time_progress()
{
    if (timeId >= tlag)
    {
        if ((timeId - tlag) % (int)(nt / 10) == 0) 
        {
            show_information();
            
            int percent = (int)floorf((float)(timeId - tlag + 1) / (float)(nt) * 100.0f);  
            
            std::cout << "\nPropagation progress: " << percent << " % \n";
        }   
    }
}

void Modeling::compute_seismogram()
{
    compute_seismogram_GPU<<<sBlocks,NTHREADS>>>(d_P, d_rIdx, d_rIdz, d_rkwPs, d_seismogram_Ps, max_spread, timeId, tlag, nt, nzz);     
    compute_seismogram_GPU<<<sBlocks,NTHREADS>>>(d_Vx, d_rIdx, d_rIdz, d_rkwVx, d_seismogram_Vx, max_spread, timeId, tlag, nt, nzz);     
    compute_seismogram_GPU<<<sBlocks,NTHREADS>>>(d_Vz, d_rIdx, d_rIdz, d_rkwVz, d_seismogram_Vz, max_spread, timeId, tlag, nt, nzz);     
}

void Modeling::export_seismogram()
{   
    cudaMemcpy(h_seismogram_Ps, d_seismogram_Ps, nt*max_spread*sizeof(float), cudaMemcpyDeviceToHost);    
    cudaMemcpy(h_seismogram_Vx, d_seismogram_Vx, nt*max_spread*sizeof(float), cudaMemcpyDeviceToHost);    
    cudaMemcpy(h_seismogram_Vz, d_seismogram_Vz, nt*max_spread*sizeof(float), cudaMemcpyDeviceToHost);    

    std::string seismPs = seismogram_folder + modeling_type + "_Ps_nStations" + std::to_string(geometry->spread[srcId]) + "_nSamples" + std::to_string(nt) + "_shot_" + std::to_string(geometry->sInd[srcId]+1) + ".bin";
    std::string seismVx = seismogram_folder + modeling_type + "_Vx_nStations" + std::to_string(geometry->spread[srcId]) + "_nSamples" + std::to_string(nt) + "_shot_" + std::to_string(geometry->sInd[srcId]+1) + ".bin";
    std::string seismVz = seismogram_folder + modeling_type + "_Vz_nStations" + std::to_string(geometry->spread[srcId]) + "_nSamples" + std::to_string(nt) + "_shot_" + std::to_string(geometry->sInd[srcId]+1) + ".bin";

    export_binary_float(seismPs, h_seismogram_Ps, nt*max_spread);    
    export_binary_float(seismVx, h_seismogram_Vx, nt*max_spread);    
    export_binary_float(seismVz, h_seismogram_Vz, nt*max_spread);    
}

void Modeling::expand_boundary(float * input, float * output)
{
    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        int i = (int) (index % nz);  
        int j = (int) (index / nz);    

        output[(i + nb) + (j + nb)*nzz] = input[i + j*nz];     
    }

    for (int i = 0; i < nb; i++)
    {
        for (int j = nb; j < nxx - nb; j++)
        {
            output[i + j*nzz] = output[nb + j*nzz];
            output[(nzz - i - 1) + j*nzz] = output[(nzz - nb - 1) + j*nzz];
        }
    }

    for (int i = 0; i < nzz; i++)
    {
        for (int j = 0; j < nb; j++)
        {
            output[i + j*nzz] = output[i + nb*nzz];
            output[i + (nxx - j - 1)*nzz] = output[i + (nxx - nb - 1)*nzz];
        }
    }
}

void Modeling::reduce_boundary(float * input, float * output)
{
    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        int i = (int) (index % nz);  
        int j = (int) (index / nz);    

        output[i + j*nz] = input[(i + nb) + (j + nb)*nzz];
    }
}

void Modeling::show_information()
{
    auto clear = system("clear");
    
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << "                                 \033[34mWASMEM2D\033[0;0m\n";
    std::cout << "-------------------------------------------------------------------------------\n\n";

    std::cout << "Model dimensions: (z = " << (nz - 1)*dz << ", x = " << (nx - 1) * dx <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << ") m\n\n";

    std::cout << "Modeling type: " << modeling_name << "\n";
}

void Modeling::compression(float * input, uintc * output, int N, float &max_value, float &min_value)
{
    max_value =-1e20f;
    min_value = 1e20f;
    
    # pragma omp parallel for
    for (int index = 0; index < N; index++)
    {
        min_value = std::min(input[index], min_value);
        max_value = std::max(input[index], max_value);        
    }

    # pragma omp parallel for
    for (int index = 0; index < N; index++)
        output[index] = static_cast<uintc>(1.0f + (float)(COMPRESS - 1)*(input[index] - min_value) / (max_value - min_value));
}

__global__ void time_set(float * T, int matsize)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < matsize) T[index] = 1e6f;
}

__global__ void time_init(float * T, float * S, float sx, float sz, float dx, 
                          float dz, int sIdx, int sIdz, int nzz, int nb)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    int zi = sIdz + (i - 1);
    int xi = sIdx + (j - 1);

    int index = zi + xi*nzz;

    T[index] = S[index] * sqrtf(powf((xi - nb)*dx - sx, 2.0f) + 
                                powf((zi - nb)*dz - sz, 2.0f));
}

__global__ void inner_sweep(float * T, float * S, int * sgnv, int * sgnt, int sgni, int sgnj, 
                            int x_offset, int z_offset, int xd, int zd, int nxx, int nzz, 
                            float dx, float dz, float dx2i, float dz2i)
{
    int element = blockIdx.x*blockDim.x + threadIdx.x;

    int i = z_offset + zd*element;
    int j = x_offset + xd*element;

    float Sref, t1, t2, t3;  

    if ((i > 0) && (i < nzz - 1) && (j > 0) && (j < nxx - 1))
    {
        int i1 = i - sgnv[sgni];
        int j1 = j - sgnv[sgnj];

        float tv = T[i - sgnt[sgni] + j*nzz];
        float te = T[i + (j - sgnt[sgnj])*nzz];
        float tev = T[(i - sgnt[sgni]) + (j - sgnt[sgnj])*nzz];

        float t1d1 = tv + dz*min(S[i1 + max(j - 1, 1)*nzz], S[i1 + min(j, nxx - 1)*nzz]); 
        float t1d2 = te + dx*min(S[max(i - 1, 1) + j1*nzz], S[min(i, nzz - 1) + j1*nzz]); 

        float t1D = min(t1d1, t1d2);

        t1 = t2 = t3 = 1e6f; 

        Sref = S[i1 + j1*nzz];

        if ((tv <= te + dx*Sref) && (te <= tv + dz*Sref) && (te - tev >= 0.0f) && (tv - tev >= 0.0f))
        {
            float ta = tev + te - tv;
            float tb = tev - te + tv;

            t1 = ((tb*dz2i + ta*dx2i) + sqrtf(4.0f*Sref*Sref*(dz2i + dx2i) - dz2i*dx2i*(ta - tb)*(ta - tb))) / (dz2i + dx2i);
        }
        else if ((te - tev <= Sref*dz*dz / sqrtf(dx*dx + dz*dz)) && (te - tev > 0.0f))
        {
            t2 = te + dx*sqrtf(Sref*Sref - ((te - tev) / dz)*((te - tev) / dz));
        }    
        else if ((tv - tev <= Sref*dx*dx / sqrt(dx*dx + dz*dz)) && (tv - tev > 0.0f))
        {
            t3 = tv + dz*sqrtf(Sref*Sref - ((tv - tev) / dx)*((tv - tev) / dx));
        }    

        float t2D = min(t1, min(t2, t3));

        T[i + j*nzz] = min(T[i + j*nzz], min(t1D, t2D));
    }
}

__global__ void compute_seismogram_GPU(float * WF, int * rIdx, int * rIdz, float * rkw, float * seismogram, int spread, int tId, int tlag, int nt, int nzz)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < spread) && (tId > tlag))
    {   
        seismogram[(tId - tlag) + index*nt] = 0.0f;    
                
        for (int i = 0; i < DGS; i++)
        {
            int zi = rIdz[index] + i - 2;

            for (int j = 0; j < DGS; j++)
            {
                int xi = rIdx[index] + j - 2;

                seismogram[(tId - tlag) + index*nt] += rkw[i + j*DGS + index*DGS*DGS]*WF[zi + xi*nzz];
            }
        }
    }    
}

__device__ float get_boundary_damper(float * d1D, float * d2D, int i, int j, int nxx, int nzz, int nb)
{
    float damper;

    // global case
    if ((i >= nb) && (i < nzz - nb) && (j >= nb) && (j < nxx - nb))
    {
        damper = 1.0f;
    }

    // 1D damping
    else if ((i >= 0) && (i < nb) && (j >= nb) && (j < nxx - nb)) 
    {
        damper = d1D[i];
    }         
    else if ((i >= nzz - nb) && (i < nzz) && (j >= nb) && (j < nxx - nb)) 
    {
        damper = d1D[nb - (i - (nzz - nb)) - 1];
    }         
    else if ((i >= nb) && (i < nzz - nb) && (j >= 0) && (j < nb)) 
    {
        damper = d1D[j];
    }
    else if ((i >= nb) && (i < nzz - nb) && (j >= nxx - nb) && (j < nxx)) 
    {
        damper = d1D[nb - (j - (nxx - nb)) - 1];
    }

    // 2D damping 
    else if ((i >= 0) && (i < nb) && (j >= 0) && (j < nb))
    {
        damper = d2D[i + j*nb];
    }
    else if ((i >= nzz - nb) && (i < nzz) && (j >= 0) && (j < nb))
    {
        damper = d2D[nb - (i - (nzz - nb)) - 1 + j*nb];
    }
    else if((i >= 0) && (i < nb) && (j >= nxx - nb) && (j < nxx))
    {
        damper = d2D[i + (nb - (j - (nxx - nb)) - 1)*nb];
    }
    else if((i >= nzz - nb) && (i < nzz) && (j >= nxx - nb) && (j < nxx))
    {
        damper = d2D[nb - (i - (nzz - nb)) - 1 + (nb - (j - (nxx - nb)) - 1)*nb];
    }

    return damper;
}
