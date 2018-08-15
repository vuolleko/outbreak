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
#include <math.h>
#include <random>
#include <vector>
#include <cstdlib>
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
        uint n_output = lrint(1. * params.n_iter / params.output_interval);

        this->counters = Eigen::MatrixXi::Zero(n_output, N_STATES);

        std::vector<Infectee *> new_infected, new_infected1;
        this->infected.push_back(new Infectee(NULL, 0, prng, params));
        uint output_counter = 0;
        for (uint time = 1; time <= params.n_iter; ++time)
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

                if (time % params.output_interval == 0)
                    this->counters(output_counter, (*it)->istatus())++;
            }

            if (!new_infected.empty()) // append all new infectees from time step
            {
                this->infected.reserve(this->infected.size() + new_infected.size());
                this->infected.insert(this->infected.end(), new_infected.begin(), new_infected.end());
                // std::cout << "t=" << time << ": New infected " << new_infected.size() << ", total " << infected.size() << std::endl;
                new_infected.clear();
            }

            if (time % params.output_interval == 0)
            {
                std::cout << "t=" << time << ": " << this->counters.row(output_counter) << std::endl;
                output_counter++;
            }

            if (this->infected.size() > params.max_infected)
            {
                std::cout << "Max number of infected individuals reached. Stopping." << std::endl;
                break;
            }
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
};

int main(int argc, char *argv[])
{
    params_struct params;
    // params.n_iter = 70;
    uint seed;
    double R0;

    if (argc > 1)
    {
        seed = std::atoi(argv[1]);
    }
    else
    {
        seed = 0;
    }
    if (argc > 2)
    {
        R0 = std::atof(argv[2]);
    }
    else
    {
        R0 = 1.7;
    }
    std::mt19937_64 prng(seed);
    params.infect_delta = params.infect_period_shape * params.infect_period_scale / R0;

    Outbreak ob(prng, params);

    std::cout << "Estimated R0: " << ob.getR0() << std::endl;

    // Eigen::MatrixXi c = ob.getCounters();
    // std::cout << c << std::endl;

    std::vector<Infectee*> inf = ob.getInfected();
    std::cout << *(inf[0]) << std::endl;
    std::cout << *(inf[2]) << std::endl;
    std::cout << *(inf[2]) << std::endl;

    return 0;
}
