// Contains the implementation for an infected individual.

#include <iostream>
#include <math.h>
#include <random>

#include "infectee.hpp"

Infectee::Infectee(Infectee *infector, double infection_time, std::mt19937_64 &prng, params_struct params) : infector(infector), infection_time(infection_time)
{

    // setup random distributions
    std::gamma_distribution<double> gamma_latent_period(params.latent_period_shape,
                                                        params.latent_period_scale);
    std::uniform_real_distribution<double> unif_incub_factor(params.incub_factor_min,
                                                             params.incub_factor_max);
    std::gamma_distribution<double> gamma_infect_period(params.infect_period_shape,
                                                        params.infect_period_scale);
    std::bernoulli_distribution will_recover(params.p_recovery);
    this->rInfect = std::bernoulli_distribution(params.timestep / params.infect_delta);

    // In the following several lines, set future evolution steps of the infection
    this->status_trajectory.push_back(0);
    this->end_times = Eigen::ArrayXd(N_STATES);
    this->end_times = std::nan("1."); // default times NaNs
    double latent_period = gamma_latent_period(prng);
    double incubation_factor = unif_incub_factor(prng);

    // incubation time may differ from latent time
    if (incubation_factor > 1.)
    {
        this->status_trajectory.push_back(1);
        this->end_times[0] = latent_period;
        this->end_times[1] = incubation_factor * latent_period;
    }
    else
    { // symptoms after infectious
        this->status_trajectory.push_back(2);
        this->end_times[0] = incubation_factor * latent_period;
        this->end_times[2] = latent_period;
    }

    this->status_trajectory.push_back(3);
    double infectious_period = gamma_infect_period(prng);
    double two_periods = latent_period + infectious_period;
    this->end_times[3] = two_periods;

    double time_end;
    if (will_recover(prng))
    {
        this->status_trajectory.push_back(4);
        this->status_trajectory.push_back(6);
        std::gamma_distribution<double> gamma_recover_period(params.recover_period_shape,
                                                             params.recover_period_scale);
        double recover_period = gamma_recover_period(prng);
        time_end = two_periods + recover_period;
    }
    else
    {
        this->status_trajectory.push_back(5);
        this->status_trajectory.push_back(7);
        std::gamma_distribution<double> gamma_dying_period(params.dying_period_shape,
                                                           params.dying_period_scale);
        double dying_period = gamma_dying_period(prng);
        time_end = two_periods + dying_period;
    }
    this->end_times[this->status_trajectory[3]] = time_end;
    this->end_times = this->end_times + infection_time;

    this->status_iter = this->status_trajectory.begin();
    this->time_last_infection = std::nan("1.");
}

Infectee::~Infectee()
{
    // std::cout << "Infectee destroyed" << std::endl;
}

Infectee *Infectee::infect(Infectee *other)
{
    // Mark `other` as infected by self.
    this->infected.push_back(other);
    return other;
}

int Infectee::n_infected() const
{
    // Return the number of infected by self.
    return this->infected.size();
}

int Infectee::istatus() const
{
    // Return the index to current status;
    return *(this->status_iter);
}

std::string Infectee::status() const
{
    // Return current status from the State enum.
    return States[this->istatus()];
}

bool Infectee::can_infect() const
{
    // Return whether self can infect others.
    return (this->istatus() == 2) || (this->istatus() == 3);
}

bool Infectee::is_reported() const
{
    // Return whether infection has been reported (i.e. not latent).
    return (this->istatus() > 2) || (this->istatus() == 1);
}

double Infectee::time_next() const
{
    // Return time of next phase in infection.
    return this->end_times[this->istatus()];
}

std::vector<Infectee *> Infectee::update(double time, std::mt19937_64 &prng, params_struct params)
{
    // Depending on time, update status of infection and possibly infect someone.
    std::vector<Infectee *> new_infected;  // Currently size<=1

    bool infectious = this->can_infect();

    while (time >= this->time_next())
    {
        this->status_iter++;
        infectious = infectious || this->can_infect();
    }

    if (infectious)
    {
        if (this->rInfect(prng))
        {
            this->time_last_infection = time;
            new_infected.push_back(this->infect(new Infectee(this, time, prng, params)));
        }
    }

    return new_infected;
}
