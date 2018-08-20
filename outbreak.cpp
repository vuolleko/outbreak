/*
Simulate outbreak of infectious disease.

Infected individuals infect others from an infinite pool. The model keeps track of
who infected whom and when. Infected individuals are initially in a latent phase i.e.
they show no symptoms nor can infect others. The illness then progresses according to
stochastic processes.

Follows the model description in:

Tom Britton and Gianpaolo Scalia Tomba (2018)
Estimation in emerging epidemics: biases and remedies, arXiv:1803.01688v1.
*/

#include <iostream>
#include <iomanip>
#include <math.h>
#include <random>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <Eigen/Core>

#include "infectee.hpp"

class Outbreak
{
  public:
    std::vector<Infectee *> infected; // infected individuals (present and past)
    Eigen::MatrixXi counters;         // counts of each infection state per output interval
    std::mt19937_64 prng;             // pseudo random-number generator
    params_struct params;             // user-given parameters (defaults in infectee.hpp)

    Outbreak(std::mt19937_64 &prng, const params_struct &params = params_struct()) : prng(prng), params(params)

    {
        uint n_output = lrint(1. * params.max_time / params.output_interval);
        this->counters = Eigen::MatrixXi::Zero(n_output, N_STATES);

        std::vector<Infectee *> new_infected, new_infected1;
        this->infected.push_back(new Infectee(NULL, 0, prng, params));
        uint output_counter = 0;

        double time = params.timestep;
        while (time <= params.max_time)
        {
            // iterate over all infected individuals
            for (std::vector<Infectee *>::iterator it = this->infected.begin(); it != this->infected.end(); ++it)
            {
                new_infected1 = (*it)->update(time, prng, params);

                if (!new_infected1.empty()) // append new infectees by single infector
                {
                    new_infected.reserve(new_infected.size() + new_infected1.size());
                    new_infected.insert(new_infected.end(), new_infected1.begin(), new_infected1.end());
                }

                if (std::fmod(time, params.output_interval) < params.timestep)
                    this->counters(output_counter, (*it)->istatus())++;
            }

            if (!new_infected.empty()) // append all new infectees from time step
            {
                this->infected.reserve(this->infected.size() + new_infected.size());
                this->infected.insert(this->infected.end(), new_infected.begin(), new_infected.end());
                // std::cout << "t=" << time << ": New infected " << new_infected.size() << ", total " << infected.size() << std::endl;
                new_infected.clear();
            }

            if (std::fmod(time, params.output_interval) < params.timestep)
            {
                if (params.verbose)
                    std::cout << "t=" << time << ": " << this->counters.row(output_counter) << std::endl;
                output_counter++;
            }

            if (this->infected.size() > params.max_infected)
            {
                if (params.verbose)
                    std::cout << "Max number of infected individuals reached. Stopping." << std::endl;
                break;
            }
            time += params.timestep;
        }
    }

    ~Outbreak()
    {
        for (std::vector<Infectee *>::iterator it = this->infected.begin(); it != this->infected.end(); ++it)
            delete *it;  // need to release these manually as allocated dynamically
    }

    Eigen::MatrixXi getCounters()
    {
        return this->counters;
    }

    std::vector<Infectee*> getInfected()
    {
        return this->infected;
    }

    float getR0()
    {
        // Estimate the basic reproduction number (R0) by considering
        // reported cases due to infectors now past the infectious period.
        int n_infected = 0;
        int n_infectors = 0;

        for (std::vector<Infectee *>::iterator it = this->infected.begin(); it != this->infected.end(); ++it)
        {
            if ((*it)->istatus() > 3)
            {
                n_infectors++;
                for (std::vector<Infectee *>::iterator it2 = (*it)->infected.begin(); it2 != (*it)->infected.end(); ++it2)
                {
                    if ((*it2)->is_reported())
                        n_infected++;
                }
            }
        }

        return (float) n_infected / n_infectors;
    }

    // Print various statistics for debugging.
    void printStats()
    {
        const uint N_GROUPS = 4;
        Eigen::ArrayXd end_time_sums = Eigen::ArrayXd::Zero(N_GROUPS);
        Eigen::ArrayXi status_sums = Eigen::ArrayXi::Zero(N_GROUPS);
        double offset;

        for (std::vector<Infectee *>::iterator it = this->infected.begin(); it != this->infected.end(); ++it)
        {
            // handle latent period
            if ((*it)->status_trajectory[1] == 1)
                offset = (*it)->end_times[0];
            else
                offset = (*it)->end_times[2];
            end_time_sums[0] += offset - (*it)->infection_time;
            status_sums[0]++;

            // infectious period
            end_time_sums[1] += (*it)->end_times[3] - offset;
            offset = (*it)->end_times[3];
            status_sums[1]++;

            // recovering period
            if ((*it)->status_trajectory[3] == 4)
            {
                end_time_sums[2] += (*it)->end_times[4] - offset;
                status_sums[2]++;
            }
            else // dying period
            {
                end_time_sums[3] += (*it)->end_times[5] - offset;
                status_sums[3]++;
            }
        }

        std::cout.precision(5);
        std::cout << std::setw(20) << "Means:" << std::setw(20) << "Latent period" 
                  << std::setw(20) << "Infectious period" << std::setw(20) << "Recovering period" 
                  << std::setw(20) << "Dying period" << std::endl;
        std::cout << std::setw(20) << (end_time_sums / status_sums.cast<double>()).transpose() << std::endl;
        std::cout << std::setw(20) << "Expected:" << std::setw(20) << params.latent_period_scale * params.latent_period_shape 
                  << std::setw(20) << params.infect_period_scale * params.infect_period_shape
                  << std::setw(20) << params.recover_period_scale * params.recover_period_shape
                  << std::setw(20) << params.dying_period_scale * params.dying_period_shape << std::endl;
        std::cout << "Pr(recovery): " << (1. * status_sums[2]) / (status_sums[2] + status_sums[3]) 
                  << " Expected " << params.p_recovery << std::endl;
    }
};

int main(int argc, char *argv[])
{
    params_struct params;
    params.verbose = true;
    uint seed;
    double R0;

    if (argc > 1)
    {
        R0 = std::atof(argv[1]);
    }
    else
    {
        R0 = 1.7;
    }
    if (argc > 2)
    {
        seed = std::atoi(argv[2]);
    }
    else
    {
        seed = static_cast<uint>(std::chrono::system_clock::now().time_since_epoch().count());
        std::cout << "Using seed = " << seed << std::endl;
    }
    std::mt19937_64 prng(seed);
    params.infect_delta = params.infect_period_shape * params.infect_period_scale / R0;

    Outbreak ob(prng, params);

    std::cout << "Estimated R0: " << ob.getR0() << std::endl;

    // Eigen::MatrixXi c = ob.getCounters();
    // std::cout << c << std::endl;

    std::vector<Infectee*> inf = ob.getInfected();
    if (inf.size() > 3)
    {
        std::cout << *(inf[0]) << std::endl;
        std::cout << *(inf[1]) << std::endl;
        std::cout << *(inf[2]) << std::endl;
    }

    ob.printStats();
    return 0;
}
