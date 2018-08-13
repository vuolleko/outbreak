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
                // std::cout << "Infected total " << new_infected.size() << std::endl;
                this->infected.reserve(this->infected.size() + new_infected.size());
                this->infected.insert(this->infected.end(), new_infected.begin(), new_infected.end());
                new_infected.clear();
            }

            if (time % params.output_interval == 0)
                output_counter++;
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
};

int main(int argc, char *argv[])
{
    params_struct params;
    params.n_iter = 70;
    uint seed;

    if (argc > 1)
    {
        seed = std::atoi(argv[1]);
    }
    else
    {
        seed = 0;
    }
    std::mt19937_64 prng(seed);

    Outbreak ob(prng, params);
    Eigen::MatrixXi c = ob.getCounters();

    std::cout << c << std::endl;

    std::vector<Infectee*> inf = ob.getInfected();
    std::cout << *(inf[0]) << std::endl;
    std::cout << *(inf[2]) << std::endl;
    std::cout << *(inf[2]) << std::endl;

    return 0;
}
