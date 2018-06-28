/*
Simulate outbreak of infectious disease.

Infected individuals infect others from an infinite pool. The model keeps track of
who infected whom and when. Infected individuals are initially in a latent phase i.e.
they show no symptoms nor can infect others. The illness then progresses according to
stochastic processes.

The current model setup is intended for inferring the basic reproduction number R0.

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

Eigen::MatrixXi simulate(double R0, std::mt19937_64 &prng, const params_struct &params = params_struct())
{

    // double mean_inf_period = params.infect_period_shape * params.infect_period_scale;
    // params.infect_delta = mean_inf_period / R0;

    uint n_output = lrint(1. * params.n_iter / params.output_interval);

    Eigen::MatrixXi counters = Eigen::MatrixXi::Zero(n_output, N_STATES);

    std::vector<Infectee *> infected, new_infected, new_infected1;
    infected.push_back(new Infectee(NULL, 0, prng, params));
    uint output_counter = 0;
    for (int time = 1; time <= params.n_iter; ++time)
    {
        // iterate over all infected individuals
        for (std::vector<Infectee *>::iterator it = infected.begin(); it != infected.end(); ++it)
        {
            new_infected1 = (*it)->update(time, prng, params);

            if (!new_infected1.empty()) // append new infectees by single infector
            {
                new_infected.reserve(new_infected.size() + new_infected1.size());
                new_infected.insert(new_infected.end(), new_infected1.begin(), new_infected1.end());
            }

            if (time % params.output_interval == 0)
                counters(output_counter, (*it)->istatus())++;
        }

        if (!new_infected.empty()) // append all new infectees from time step
        {
            // std::cout << "Infected total " << new_infected.size() << std::endl;
            infected.reserve(infected.size() + new_infected.size());
            infected.insert(infected.end(), new_infected.begin(), new_infected.end());
            new_infected.clear();
        }

        if (time % params.output_interval == 0)
            output_counter++;
    }

    // std::cout << *(infected[0]) << std::endl;
    // std::cout << *(infected[1]) << std::endl;
    // std::cout << *(infected[2]) << std::endl;
    return counters;
}

int main(int argc, char *argv[])
{
    double R0 = 1.7;
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

    Eigen::MatrixXi c = simulate(R0, prng, params);

    std::cout << c << std::endl;

    return 0;
}
