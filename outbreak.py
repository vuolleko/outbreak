"""Outbreak of infectious disease."""

import numpy as np
import scipy.stats as ss


status_dict = {i: v for i, v in enumerate(
    ['latent', 'symptoms-non-infectious', 'latent-infectious', 'symptoms',
     'recovering', 'dying', 'recovered', 'dead'])}
n_states = len(status_dict)


def simulate(R0, full_output=False, random_state=np.random, **kwargs):
    """Simulate outbreak of infectious disease.

    Infected individuals infect others from an infinite pool. The model keeps track of
    who infected whom and when. Infected individuals are initially in a latent phase i.e.
    they show no symptoms nor can infect others. The illness then progresses according to
    stochastic processes.

    The current model setup is intended for inferring the basic reproduction number R0.

    Follows the model description in:

    Tom Britton and Gianpaolo Scalia Tomba (2018)
    Estimation in emerging epidemics: biases and remedies, arXiv:1803.01688v1.

    Parameters
    ----------
    R0 : float
        Basic reproduction number i.e. the mean number of infections during the infectious period.
    full_output : bool, optional
        Whether to return a pd.DataFrame with status of each step (very slow).
    random_state : np.random.RandomState, optional
    kwargs
        Model parameters may be given as keyword arguments. See code.

    Returns
    -------
    pd.DataFrame if full_output else np.array

    """
    params = {'latent_period': {'a': 2., 'scale': 5.},  # gamma
              'incub_factor': {'loc': 0.8, 'scale': 0.4},  # uniform
              'infectious_period': {'a': 1., 'scale': 5.},  # gamma
              'will_recover': {'p': 0.3},  # bernoulli
              'recover_period': {'a': 4., 'scale': 3.},  # gamma
              'dying_period': {'a': 4./9., 'scale': 9.},  # gamma
              'n_iter': 154,  # number of model iterations (e.g. days)
              'output_freq': 7}  # frequency of output (e.g. week)
    params.update(kwargs)

    # convert R0 into a period between infections caused by single individual
    mean_inf_period = params['infectious_period']['a'] * params['infectious_period']['scale']
    infect_delta = mean_inf_period / R0
    params['infect_delta'] = infect_delta

    time = 0
    infected = [Infectee(None, time, params, random_state)]  # start with 1

    n_output = round(params['n_iter'] / params['output_freq'] + 0.49999)
    counters = np.zeros((n_output, n_states), dtype=np.int32)
    i_counter = 0

    while time < params['n_iter']:
        time += 1

        for i in infected:
            new_i = i.update(time, params, random_state)
            if new_i:  # new infectee
                infected.append(new_i)

            if time % params['output_freq'] == 0:
                counters[i_counter, i.istatus] += 1

        if time % params['output_freq'] == 0:
            i_counter += 1

    return counters


class Infectee:
    """Infected individual, instantiated upon infection.

    Attributes
    ----------
    infector : Infectee
        The individual who caused infection.
    infection_time : int
        Time of infection.
    infected : list
        Individuals infected by self.
    status_trajectory : list
        Progression of infection with respect to `status_dict`.
    end_times : np.array
        End times of phases in `status_trajectory`.

    """

    def __init__(self, infector, infection_time, params, random_state=None):
        """Create a new infected individual.

        Parameters
        ----------
        infector : Infectee
            The individual who caused infection.
        infection_time : int
            Time of infection.
        params : dict
            A number of model parameters. See code.
        random_state : np.random.RandomState, optional

        """
        self.infector = infector  # individual who infected self
        self.infected = []  # list of individuals infected by self
        self.infection_time = infection_time
        self._last_infection = infection_time  # counter starts

        # set future evolution steps of the infection
        self.status_trajectory = [0]  # ref. `status_dict`
        self.end_times = np.full(n_states, np.nan)  # unit: 1 day
        latent_period = ss.gamma.rvs(**params['latent_period'], random_state=random_state)
        incubation_factor = ss.uniform.rvs(**params['incub_factor'], random_state=random_state)

        # incubation time may differ from latent time
        if incubation_factor > 1:
            self.status_trajectory.append(1)
            self.end_times[0] = latent_period
            self.end_times[1] = incubation_factor * latent_period
        else:  # symptoms before infectious
            self.status_trajectory.append(2)
            self.end_times[0] = incubation_factor * latent_period
            self.end_times[2] = latent_period

        self.status_trajectory.append(3)
        infectious_period = ss.gamma.rvs(**params['infectious_period'], random_state=random_state)
        two_periods = latent_period + infectious_period
        self.end_times[3] = two_periods

        will_recover = ss.bernoulli.rvs(**params['will_recover'], random_state=random_state)
        if will_recover:
            self.status_trajectory.extend([4, 6])
            recover_period = ss.gamma.rvs(**params['recover_period'], random_state=random_state)
            time_end = two_periods + recover_period
        else:
            self.status_trajectory.extend([5, 7])
            dying_period = ss.gamma.rvs(**params['dying_period'], random_state=random_state)
            time_end = two_periods + dying_period
        self.end_times[self.status_trajectory[-2]] = time_end
        self.end_times += infection_time

        self._status_iter = iter(self.status_trajectory)
        self.istatus = next(self._status_iter)

    def infect(self, other):
        """Mark `other` as infected by self.

        Parameters
        ----------
        other : Infectee

        """
        self.infected.append(other)
        return other

    @property
    def n_infected(self):
        """Return the number of infected by self."""
        return len(self.infected)

    @property
    def can_infect(self):
        """Return whether self can infect others."""
        return (self.istatus == 2) or (self.istatus == 3)

    @property
    def status(self):
        """Return a string representation of current infection status."""
        return status_dict[self.istatus]

    @property
    def _time_next(self):
        """Return time of next phase in infection."""
        return self.end_times[self.istatus]

    def update(self, time, params, random_state=None):
        """Depending on time, update status of infection and infect someone.

        Parameters
        ----------
        time : int
            Current time.
        params : dict
            Model parameters. See code.
        random_state : np.random.RandomState, optional

        Returns
        -------
        Infectee or None

        """
        if time >= self._time_next:
            self.istatus = next(self._status_iter)

        if self.can_infect:
            if time - self._last_infection > params['infect_delta']:
                new_infectee = self.infect(Infectee(self, time, params, random_state))
                self._last_infection = time

                return new_infectee
        return None

    def __str__(self):
        """Return string representation of self."""
        str = "Individual {} has infected {} others: {}".format(
            hex(id(self)), self.n_infected, [hex(id(i)) for i in self.infected])
        return str


def print_summary(counters):
    """Summarize final result from simulation."""
    for i in range(n_states):
        print("{}: {}".format(status_dict[i], counters[-1, i]))
    print("\nCases: {} reported, {} unreported, {} total".format(
          n_observed(counters[-1]), n_latent(counters[-1]), n_cases(counters[-1])))


def n_cases(counters):
    """Return total number of cases (reported and unreported)."""
    return np.sum(counters, axis=-1)


def n_observed(counters):
    """Return total number of observed cases."""
    return np.sum(counters[..., [1, 3, 4, 5, 6, 7]], axis=-1)


def n_latent(counters):
    """Return total number of unobserved cases."""
    return np.sum(counters[..., [0, 2]], axis=-1)


if __name__ == '__main__':
    seed = 2
    R0 = 1.7

    counters = simulate(R0, random_state=np.random.RandomState(seed))
    print_summary(counters)
