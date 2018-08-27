#ifndef INFECTEE_H
#define INFECTEE_H

#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Core>

typedef unsigned int uint;

const uint N_STATES = 8; // number of different infection statuses

// Default settings for simulation
// https://softwareengineering.stackexchange.com/a/329733
struct params_struct
{
    double latent_period_shape = 2.; // gamma
    double latent_period_scale = 5.;
    double incub_factor_min = 0.8; // uniform
    double incub_factor_max = 1.2;
    double infect_period_shape = 1.; // gamma
    double infect_period_scale = 5.;
    double p_recovery = 0.3;          // bernoulli
    double recover_period_shape = 4.; // gamma
    double recover_period_scale = 3.;
    double dying_period_shape = 4. / 9.; // gamma
    double dying_period_scale = 9.;
    double infect_delta = 2.941; // avg time between infections
    double max_time = 364.;           // max model time (e.g. days)
    double output_interval = 7.;    // interval of output (e.g. week)
    double timestep = 0.2;
    uint max_infected = 100000;  // stop iterating if reached
    bool verbose = false;  // true for printing progress etc.
};

// Infection states (ref. Infection.istatus)
const std::string States[N_STATES]{
    "latent",
    "symptoms_non_infectious",
    "latent_infectious",
    "symptoms",
    "recovering",
    "dying",
    "recovered",
    "dead"};

class Infectee
{
    public:
        Infectee(Infectee *infector, double infection_time, std::mt19937_64 &prng, params_struct params);
        ~Infectee();

        bool can_infect() const;           // Return whether self can infect others.
        bool is_reported() const;          // Return whether infection has been reported.
        std::string status() const;        // Return current status from the State enum.

        std::vector<Infectee *> update(double time, std::mt19937_64 &prng, params_struct params); // Depending on time, update status of infection and possibly infect someone.

    private:
        const Infectee *infector;          // The individual who caused infection.
        const double infection_time;       // Time of infection.

        Infectee *infect(Infectee *other); // Mark `other` as infected by self.
        std::vector<Infectee *> infected;  // Individuals infected by self.
        int n_infected() const;            // Return the number of infected by self.

        std::vector<uint> status_trajectory;     // Progression of infection with respect to infection states.
        Eigen::ArrayXd end_times;                // End times of phases in `status_trajectory`.
        std::vector<uint>::iterator status_iter; // Iterator for `status_trajectory`.

        int istatus() const;               // Return the index to current status;
        double time_next() const;          // Return time of next phase in infection.
        double time_last_infection;        // Time of latest infection by self.

        std::bernoulli_distribution rInfect;  // random engine for infecting

    friend class Outbreak;
    friend std::ostream &operator<<(std::ostream &os, Infectee const &inf);
};

// Allow printing a representation of Infectee objects
inline std::ostream &operator<<(std::ostream &os, Infectee const &inf)
{
    os << "Individual " << &inf << " was infected at t=" << inf.infection_time;
    os << " and has infected " << inf.n_infected() << " others: ";
    for (int i = 0; i < inf.n_infected(); ++i)
    {
        os << ' ' << inf.infected[i];
        os.flush();
    }
    return os;
}

#endif