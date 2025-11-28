# include "triclinic.cuh"

void Triclinic::set_parameters()
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

    eikonalClip = str2bool(catch_parameter("eikonalClip", parameters));
    compression = str2bool(catch_parameter("compression", parameters));

    snapshot_folder = catch_parameter("snapshot_folder", parameters);
    seismogram_folder = catch_parameter("seismogram_folder", parameters);

    nPoints = nx*nz;

    nxx = nx + 2*nb;
    nzz = nz + 2*nb;

    matsize = nxx*nzz;

    nBlocks = (int)((matsize + NTHREADS - 1) / NTHREADS);

    set_models();
    set_wavelet();
    set_dampers();
    set_eikonal();
    set_geometry();
    set_snapshots();
    set_seismogram();
    set_wavefields();

    set_modeling_type();
}

void Triclinic::set_models()
{
    std::string slowness_file = catch_parameter("slowness_file", parameters);
    std::string buoyancy_file = catch_parameter("buoyancy_file", parameters);
    std::string Cijkl_folder = catch_parameter("Cijkl_folder", parameters);

    auto * iModel = new float[nPoints]();
    auto * xModel = new float[matsize]();
    auto * uModel = new uintc[matsize]();
    
    import_binary_float(slowness_file, iModel, nPoints);
    expand_boundary(iModel, xModel);
    cudaMalloc((void**)&(d_S), matsize*sizeof(float));
    cudaMemcpy(d_S, xModel, matsize*sizeof(float), cudaMemcpyHostToDevice);

    if (compression)
    {
        import_binary_float(buoyancy_file, iModel, nPoints);
        expand_boundary(iModel, xModel);
        get_compression(xModel, uModel, matsize, maxB, minB);        
        cudaMalloc((void**)&(dc_B), matsize*sizeof(uintc));
        cudaMemcpy(dc_B, uModel, matsize*sizeof(uintc), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C11.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        get_compression(xModel, uModel, matsize, maxC11, minC11);        
        cudaMalloc((void**)&(dc_C11), matsize*sizeof(uintc));
        cudaMemcpy(dc_C11, uModel, matsize*sizeof(uintc), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C13.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        get_compression(xModel, uModel, matsize, maxC13, minC13);        
        cudaMalloc((void**)&(dc_C13), matsize*sizeof(uintc));
        cudaMemcpy(dc_C13, uModel, matsize*sizeof(uintc), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C15.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        get_compression(xModel, uModel, matsize, maxC15, minC15);        
        cudaMalloc((void**)&(dc_C15), matsize*sizeof(uintc));
        cudaMemcpy(dc_C15, uModel, matsize*sizeof(uintc), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C33.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        get_compression(xModel, uModel, matsize, maxC33, minC33);        
        cudaMalloc((void**)&(dc_C33), matsize*sizeof(uintc));
        cudaMemcpy(dc_C33, uModel, matsize*sizeof(uintc), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C35.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        get_compression(xModel, uModel, matsize, maxC35, minC35);        
        cudaMalloc((void**)&(dc_C35), matsize*sizeof(uintc));
        cudaMemcpy(dc_C35, uModel, matsize*sizeof(uintc), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C55.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        get_compression(xModel, uModel, matsize, maxC55, minC55);        
        cudaMalloc((void**)&(dc_C55), matsize*sizeof(uintc));
        cudaMemcpy(dc_C55, uModel, matsize*sizeof(uintc), cudaMemcpyHostToDevice);
    }
    else 
    {
        import_binary_float(buoyancy_file, iModel, nPoints);
        expand_boundary(iModel, xModel);
        cudaMalloc((void**)&(d_B), matsize*sizeof(float));
        cudaMemcpy(d_B, xModel, matsize*sizeof(float), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C11.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        cudaMalloc((void**)&(d_C11), matsize*sizeof(float));
        cudaMemcpy(d_C11, xModel, matsize*sizeof(float), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C13.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        cudaMalloc((void**)&(d_C13), matsize*sizeof(float));
        cudaMemcpy(d_C13, xModel, matsize*sizeof(float), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C15.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        cudaMalloc((void**)&(d_C15), matsize*sizeof(float));
        cudaMemcpy(d_C15, xModel, matsize*sizeof(float), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C33.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        cudaMalloc((void**)&(d_C33), matsize*sizeof(float));
        cudaMemcpy(d_C33, xModel, matsize*sizeof(float), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C35.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        cudaMalloc((void**)&(d_C35), matsize*sizeof(float));
        cudaMemcpy(d_C35, xModel, matsize*sizeof(float), cudaMemcpyHostToDevice);

        import_binary_float(Cijkl_folder + "C55.bin", iModel, nPoints);
        expand_boundary(iModel, xModel);
        cudaMalloc((void**)&(d_C55), matsize*sizeof(float));
        cudaMemcpy(d_C55, xModel, matsize*sizeof(float), cudaMemcpyHostToDevice);
    }
}

void Triclinic::expand_boundary(float * input, float * output)
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

void Triclinic::reduce_boundary(float * input, float * output)
{
    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        int i = (int) (index % nz);  
        int j = (int) (index / nz);    

        output[i + j*nz] = input[(i + nb) + (j + nb)*nzz];
    }
}

void Triclinic::get_compression(float * input, uintc * output, int N, float &max_value, float &min_value)
{
    max_value =-1e20f;
    min_value = 1e20f;
    
    # pragma omp parallel for
    for (int index = 0; index < N; index++)
    {
        min_value = (input[index] < min_value) ? input[index] : min_value;
        max_value = (input[index] > max_value) ? input[index] : max_value;        
    }

    # pragma omp parallel for
    for (int index = 0; index < N; index++)
        output[index] = static_cast<uintc>(1.0f + (float)(COMPRESS - 1)*(input[index] - min_value) / (max_value - min_value));
}

void Triclinic::set_wavelet()
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

void Triclinic::set_dampers()
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

void Triclinic::set_eikonal()
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

    cudaMalloc((void**)&(d_sgnv), NSWEEPS*MESHDIM*sizeof(int));
    cudaMalloc((void**)&(d_sgnt), NSWEEPS*MESHDIM*sizeof(int));
    
    cudaMemcpy(d_sgnv, h_sgnv, NSWEEPS*MESHDIM*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sgnt, h_sgnt, NSWEEPS*MESHDIM*sizeof(int), cudaMemcpyHostToDevice);

    std::vector<std::vector<int>>().swap(sgnv);
    std::vector<std::vector<int>>().swap(sgnt);

    delete[] h_sgnt;
    delete[] h_sgnv;
}

void Triclinic::set_geometry()
{
    geometry = new Geometry();
    geometry->parameters = parameters;
    geometry->set_parameters();

    cudaMalloc((void**)&(d_skw), DGS*DGS*sizeof(float));

    cudaMalloc((void**)&(d_rIdx), geometry->nrec*sizeof(int));
    cudaMalloc((void**)&(d_rIdz), geometry->nrec*sizeof(int));
    
    cudaMalloc((void**)&(d_rkwPs), DGS*DGS*geometry->nrec*sizeof(float));
    //cudaMalloc((void**)&(d_rkwVx), DGS*DGS*DGS*geometry->nrec*sizeof(float));
    //cudaMalloc((void**)&(d_rkwVz), DGS*DGS*DGS*geometry->nrec*sizeof(float));

    set_rec_weights();
}

void Triclinic::set_snapshots()
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

void Triclinic::set_seismogram()
{
    sBlocks = (int)((geometry->nrec + NTHREADS - 1) / NTHREADS); 

    h_seismogram_Ps = new float[nt*geometry->nrec]();
    //h_seismogram_Vx = new float[nt*geometry->nrec]();
    //h_seismogram_Vz = new float[nt*geometry->nrec]();

    cudaMalloc((void**)&(d_seismogram_Ps), nt*geometry->nrec*sizeof(float));
    //cudaMalloc((void**)&(d_seismogram_Vx), nt*geometry->nrec*sizeof(float));
    //cudaMalloc((void**)&(d_seismogram_Vz), nt*geometry->nrec*sizeof(float));
}

void Triclinic::set_wavefields()
{
    cudaMalloc((void**)&(d_P), matsize*sizeof(float));
    cudaMalloc((void**)&(d_T), matsize*sizeof(float));

    cudaMalloc((void**)&(d_Vx), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Vz), matsize*sizeof(float));

    cudaMalloc((void**)&(d_Txx), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Tzz), matsize*sizeof(float));
    cudaMalloc((void**)&(d_Txz), matsize*sizeof(float));
}

void Triclinic::run_wave_propagation()
{
    for (srcId = 0; srcId < geometry->nsrc; srcId++)
    {
        get_shot_position();                      
        time_propagation();
        show_information();
        export_seismograms();
    }
}

void Triclinic::get_shot_position()
{
    sx = geometry->xsrc[srcId];
    sz = geometry->zsrc[srcId];

    sIdx = (int)((sx + 0.5f*dx) / dx);
    sIdz = (int)((sz + 0.5f*dz) / dz);

    set_src_weights();

    sIdx += nb;
    sIdz += nb;
}

void Triclinic::show_information()
{
    auto clear = system("clear");
    
    std::cout << "-----------------------------------------------------------\n";
    std::cout << " \033[34mEikoStagTriX2D\033[0;0m -------------------------------------------\n";
    std::cout << "-----------------------------------------------------------\n\n";

    std::cout << "Model dimensions: (z = " << (nz - 1)*dz << ", x = " << (nx - 1) * dx <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nsrc << " in total\n\n";

    std::cout << "Current shot position: (z = " << sz << ", x = " << sx << ") m\n\n";

    std::cout << "Modeling type: " << modeling_name << "\n";
}

void Triclinic::time_propagation()
{
    compute_eikonal();
    wavefield_refresh();

    if (snapshot)
    {
        snapCount = 0;

        export_travelTimes();
    }

    for (timeId = 0; timeId < nt + tlag; timeId++)
    {
        source_injection();
        compute_velocity();
        compute_pressure();
        compute_snapshots();
        compute_seismogram();    
        show_time_progress();
    }
}

void Triclinic::compute_eikonal()
{    
    if (eikonalClip)
    {
        eikonal_solver();
        
        if (compression)
        {
            uintc_quasi_slowness<<<nBlocks,NTHREADS>>>(d_T,d_S,dx,dz,sIdx,sIdz,nxx,nzz,nb,dc_C11,dc_C13,dc_C15,dc_C33,dc_C35,dc_C55,minC11,
                                                       maxC11,minC13,maxC13,minC15,maxC15,minC33,maxC33,minC35,maxC35,minC55,maxC55);        
        }
        else
        {
            float_quasi_slowness<<<nBlocks,NTHREADS>>>(d_T,d_S,dx,dz,sIdx,sIdz,nxx,nzz,nb,d_C11,d_C13,d_C15,d_C33,d_C35,d_C55);
        }
        
        eikonal_solver();
    } 
}

void Triclinic::eikonal_solver()
{
    dim3 grid(1,1,1);
    dim3 block(MESHDIM+1,MESHDIM+1,1);

    int min_level = std::min(nxx, nzz);
    int max_level = std::max(nxx, nzz);

    int z_offset, x_offset, n_elements;

    time_set<<<nBlocks,NTHREADS>>>(d_T, matsize);
    time_init<<<grid,block>>>(d_T,d_S,sx,sz,dx,dz,sIdx,sIdz,nzz,nb);

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

void Triclinic::wavefield_refresh()
{
    cudaMemset(d_P, 0.0f, matsize*sizeof(float));
    
    cudaMemset(d_Vx, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Vz, 0.0f, matsize*sizeof(float));
    
    cudaMemset(d_Txx, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Tzz, 0.0f, matsize*sizeof(float));
    cudaMemset(d_Txz, 0.0f, matsize*sizeof(float));

    cudaMemset(d_seismogram_Ps, 0.0f, nt*geometry->nrec*sizeof(float));
    cudaMemset(d_seismogram_Vx, 0.0f, nt*geometry->nrec*sizeof(float));
    cudaMemset(d_seismogram_Vz, 0.0f, nt*geometry->nrec*sizeof(float));
}

void Triclinic::export_travelTimes()
{
    if (eikonalClip)
    {
        cudaMemcpy(snapshot_in, d_T, matsize*sizeof(float), cudaMemcpyDeviceToHost);
        reduce_boundary(snapshot_in, snapshot_out);
        export_binary_float(snapshot_folder + "triclinic_eikonal_" + std::to_string(nz) + "x" + std::to_string(nx) + "_shot_" + std::to_string(srcId+1) + ".bin", snapshot_out, nPoints);    
    }
}

void Triclinic::source_injection()
{
    dim3 grid(1,1,1);
    dim3 block(DGS,DGS,1);

    if (timeId < nt) apply_pressure_source<<<grid,block>>>(d_Txx, d_Tzz, d_skw, d_wavelet, sIdx, sIdz, timeId, nzz, dx, dz);     
}

void Triclinic::compute_snapshots()
{
    if (snapshot)
    {
        if (snapCount < snapId.size())
        {
            if ((timeId-tlag) == snapId[snapCount])
            {
                cudaMemcpy(snapshot_in, d_P, matsize*sizeof(float), cudaMemcpyDeviceToHost);
                reduce_boundary(snapshot_in, snapshot_out);
                export_binary_float(snapshot_folder + modeling_type + "_snapshot_step" + std::to_string(timeId-tlag) + "_" + std::to_string(nz) + "x" + std::to_string(nx) + "_shot_" + std::to_string(srcId+1) + ".bin", snapshot_out, nPoints);    
                
                ++snapCount;
            }
        }
    }
}

void Triclinic::compute_seismogram()
{
    compute_seismogram_GPU<<<sBlocks,NTHREADS>>>(d_P, d_rIdx, d_rIdz, d_rkwPs, d_seismogram_Ps, geometry->nrec, timeId, tlag, nt, nzz);     
    //compute_seismogram_GPU<<<sBlocks,NTHREADS>>>(d_Vx, d_rIdx, d_rIdz, d_rkwVx, d_seismogram_Vx, geometry->nrec, timeId, tlag, nt, nzz);     
    //compute_seismogram_GPU<<<sBlocks,NTHREADS>>>(d_Vz, d_rIdx, d_rIdz, d_rkwVz, d_seismogram_Vz, geometry->nrec, timeId, tlag, nt, nzz);     
}

void Triclinic::show_time_progress()
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

void Triclinic::export_seismograms()
{   
    cudaMemcpy(h_seismogram_Ps, d_seismogram_Ps, nt*geometry->nrec*sizeof(float), cudaMemcpyDeviceToHost);    
    //cudaMemcpy(h_seismogram_Vx, d_seismogram_Vx, nt*geometry->nrec*sizeof(float), cudaMemcpyDeviceToHost);    
    //cudaMemcpy(h_seismogram_Vz, d_seismogram_Vz, nt*geometry->nrec*sizeof(float), cudaMemcpyDeviceToHost);    

    std::string seismPs = seismogram_folder + modeling_type + "_Ps_nStations" + std::to_string(geometry->nrec) + "_nSamples" + std::to_string(nt) + "_shot_" + std::to_string(srcId+1) + ".bin";
    //std::string seismVx = seismogram_folder + modeling_type + "_Vx_nStations" + std::to_string(geometry->nrec) + "_nSamples" + std::to_string(nt) + "_shot_" + std::to_string(srcId+1) + ".bin";
    //std::string seismVz = seismogram_folder + modeling_type + "_Vz_nStations" + std::to_string(geometry->nrec) + "_nSamples" + std::to_string(nt) + "_shot_" + std::to_string(srcId+1) + ".bin";

    export_binary_float(seismPs, h_seismogram_Ps, nt*geometry->nrec);    
    //export_binary_float(seismVx, h_seismogram_Vx, nt*geometry->nrec);    
    //export_binary_float(seismVz, h_seismogram_Vz, nt*geometry->nrec);    
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

__global__ void float_quasi_slowness(float * T, float * S, float dx, float dz, int sIdx, int sIdz, int nxx, int nzz, 
                                     int nb, float * C11, float * C13, float * C15, float * C33, float * C35, float * C55)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    const int n = 2;
    const int v = 3;

    float p[n];
    float C[v*v];
    float Gv[n];

    if ((i >= nb) && (i < nzz-nb) && (j >= nb) && (j < nxx-nb))
    {
        if (!((i == sIdz) && (j == sIdx)))    
        {
            float dTz = 0.5f*(T[(i+1) + j*nzz] - T[(i-1) + j*nzz]) / dz;
            float dTx = 0.5f*(T[i + (j+1)*nzz] - T[i + (j-1)*nzz]) / dx;

            float norm = sqrtf(dTx*dTx + dTz*dTz);

            p[0] = dTx / norm;
            p[1] = dTz / norm;
            
            C[0+0*v] = C11[index]; C[0+1*v] = C13[index]; C[0+2*v] = C15[index];
            C[1+0*v] = C13[index]; C[1+1*v] = C33[index]; C[1+2*v] = C35[index];
            C[2+0*v] = C15[index]; C[2+1*v] = C35[index]; C[2+2*v] = C55[index];

            float Ro = C33[index]*S[index]*S[index];    
            
            for (int indp = 0; indp < v*v; indp++)
                C[indp] = C[indp] / Ro / Ro;

            float Gxx = C[0+0*v]*p[0]*p[0] + C[2+2*v]*p[1]*p[1] + 2.0f*C[0+2*v]*p[0]*p[1];
            float Gzz = C[2+2*v]*p[0]*p[0] + C[1+1*v]*p[1]*p[1] + 2.0f*C[1+2*v]*p[0]*p[1];
            float Gxz = C[0+2*v]*p[0]*p[0] + C[1+2*v]*p[1]*p[1] + (C[0+1*v] + C[2+2*v])*p[0]*p[1]; 
            
            float coeff1 = Gxx + Gzz;
            float coeff2 = Gxx - Gzz;
            
            float det = sqrtf((coeff2 * coeff2) / 4.0f + Gxz * Gxz);

            Gv[0] = coeff1 / 2.0 + det;
            Gv[1] = coeff1 / 2.0 - det;
            
            if (Gv[0] < Gv[1]) {float aux = Gv[0]; Gv[0] = Gv[1]; Gv[1] = aux;} 

            S[index] = 1.0f / sqrtf(Gv[0] * Ro);
        }
    }
}

__global__ void uintc_quasi_slowness(float * T, float * S, float dx, float dz, int sIdx, int sIdz, int nxx, int nzz, 
                                     int nb, uintc * C11, uintc * C13, uintc * C15, uintc * C33, uintc * C35, uintc * C55, 
                                     float minC11, float maxC11, float minC13, float maxC13, float minC15, float maxC15, 
                                     float minC33, float maxC33, float minC35, float maxC35, float minC55, float maxC55)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    const int n = 2;
    const int v = 3;

    float p[n];
    float C[v*v];
    float Gv[n];

    if ((i >= nb) && (i < nzz-nb) && (j >= nb) && (j < nxx-nb))
    {
        if (!((i == sIdz) && (j == sIdx)))    
        {
            float dTz = 0.5f*(T[(i+1) + j*nzz] - T[(i-1) + j*nzz]) / dz;
            float dTx = 0.5f*(T[i + (j+1)*nzz] - T[i + (j-1)*nzz]) / dx;

            float norm = sqrtf(dTx*dTx + dTz*dTz);

            p[0] = dTx / norm;
            p[1] = dTz / norm;
            
            float c11 = (minC11 + (static_cast<float>(C11[index]) - 1.0f) * (maxC11 - minC11) / (COMPRESS - 1));
            float c13 = (minC13 + (static_cast<float>(C13[index]) - 1.0f) * (maxC13 - minC13) / (COMPRESS - 1));
            float c15 = (minC15 + (static_cast<float>(C15[index]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));

            float c33 = (minC33 + (static_cast<float>(C33[index]) - 1.0f) * (maxC33 - minC33) / (COMPRESS - 1));
            float c35 = (minC35 + (static_cast<float>(C35[index]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));

            float c55 = (minC55 + (static_cast<float>(C55[index]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));

            C[0+0*v] = c11; C[0+1*v] = c13; C[0+2*v] = c15;
            C[1+0*v] = c13; C[1+1*v] = c33; C[1+2*v] = c35;
            C[2+0*v] = c15; C[2+1*v] = c35; C[2+2*v] = c55;

            float Ro = c33*S[index]*S[index];    
            
            for (int indp = 0; indp < v*v; indp++)
                C[indp] = C[indp] / Ro / Ro;

            float Gxx = C[0+0*v]*p[0]*p[0] + C[2+2*v]*p[1]*p[1] + 2.0f*C[0+2*v]*p[0]*p[1];
            float Gzz = C[2+2*v]*p[0]*p[0] + C[1+1*v]*p[1]*p[1] + 2.0f*C[1+2*v]*p[0]*p[1];
            float Gxz = C[0+2*v]*p[0]*p[0] + C[1+2*v]*p[1]*p[1] + (C[0+1*v] + C[2+2*v])*p[0]*p[1]; 
            
            float coeff1 = Gxx + Gzz;
            float coeff2 = Gxx - Gzz;
            
            float det = sqrtf((coeff2 * coeff2) / 4.0f + Gxz * Gxz);

            Gv[0] = coeff1 / 2.0 + det;
            Gv[1] = coeff1 / 2.0 - det;
            
            if (Gv[0] < Gv[1]) {float aux = Gv[0]; Gv[0] = Gv[1]; Gv[1] = aux;} 

            S[index] = 1.0f / sqrtf(Gv[0] * Ro);
        }
    }
}

__global__ void apply_pressure_source(float * Txx, float * Tzz, float * skw, float * wavelet, int sIdx, int sIdz, int tId, int nzz, float dx, float dz)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    int zi = sIdz + (i - 3);
    int xi = sIdx + (j - 3);

    int index = zi + xi*nzz;
        
    Txx[index] += skw[i + j*DGS]*wavelet[tId] / (dx*dz);
    Tzz[index] += skw[i + j*DGS]*wavelet[tId] / (dx*dz);           
}

__global__ void compute_seismogram_GPU(float * WF, int * rIdx, int * rIdz, float * rkw, float * seismogram, int spread, int tId, int tlag, int nt, int nzz)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < spread) && (tId >= tlag))
    {
        // seismogram[(tId - tlag) + index*nt] = 0.0f;

        seismogram[(tId - tlag) + index*nt] = WF[rIdz[index] + rIdx[index]*nzz];

        // for (int j = 0; j < DGS; j++)
        // {
        //     int xi = rIdx[index] + j - 3;

        //     for (int i = 0; i < DGS; i++)
        //     {
        //         int zi = rIdz[index] + i - 3;

        //         seismogram[(tId - tlag) + index*nt] += rkw[i + j*DGS + index*DGS*DGS]*WF[zi + xi*nzz];
        //     }
        // }
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
