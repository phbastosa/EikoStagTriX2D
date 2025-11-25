# include <sys/resource.h>

# include "modeling/triclinic_ssg.cuh"
# include "modeling/triclinic_rsg.cuh"

int main(int argc, char **argv)
{
    std::vector<Triclinic *> modeling = 
    {
        new Triclinic_SSG(),
        new Triclinic_RSG()
    };

    auto file = std::string(argv[1]);
    auto type = std::stoi(catch_parameter("modeling_type", file));    

    modeling[type]->parameters = file;

    modeling[type]->set_parameters();

    auto ti = std::chrono::system_clock::now();

    modeling[type]->run_wave_propagation();

    auto tf = std::chrono::system_clock::now();
    
    std::chrono::duration<double> elapsed_seconds = tf - ti;
    std::cout << "\nRun time: " << elapsed_seconds.count() << " s." << std::endl;
    
    return 0;
}