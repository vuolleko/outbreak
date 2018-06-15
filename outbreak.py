"""Outbreak of infectious disease."""

import numpy as np
import pandas as pd
import scipy.stats as ss


status_dict = {i: v for i, v in enumerate(
    ['latent', 'symptoms-non-infectious', 'latent-infectious', 'symptoms',
     'recovering', 'dying', 'recovered', 'dead'])}
n_states = len(status_dict)
pd.set_option('display.width', 1000)


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
              'n_iter': 150}  # number of model iterations (e.g. days)
    params.update(kwargs)

    # convert R0 into a period between infections caused by single individual
    mean_inf_period = params['infectious_period']['a'] * params['infectious_period']['scale']
    infect_delta = mean_inf_period / R0
    params['infect_delta'] = infect_delta

    time = 0
    infected = [Infectee(None, time, params, random_state)]  # start with 1

    if full_output:
        counters = pd.DataFrame(index=np.arange(1, params['n_iter']),
                                columns=status_dict.values())

    while time < params['n_iter']:
        time += 1

        if full_output:
            counters.loc[time] = 0

        for i in infected:
            new_i = i.update(time, params, random_state)
            if new_i:
                infected.append(new_i)

            if full_output:
                counters.loc[time, i.status] += 1  # slow!!!

        if full_output and (time % 10 == 0):
            print(counters.loc[time:time])

    if not full_output:
        counters = np.zeros(n_states, dtype=np.int32)
        for i in infected:
            counters[i._status] += 1

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
        self._status = next(self._status_iter)

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
        return (self._status == 2) or (self._status == 3)

    @property
    def status(self):
        """Return a string representation of current infection status."""
        return status_dict[self._status]

    @property
    def _time_next(self):
        """Return time of next phase in infection."""
        return self.end_times[self._status]

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
            self._status = next(self._status_iter)

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


if __name__ == '__main__':
    full_output = False
    seed = 2
    R0 = 1.7

    counters = simulate(R0, full_output=False,
                        random_state=np.random.RandomState(seed))

    print("\n")
    if full_output:
        print(counters.iloc[-1])
        print("\nCases reported: {}".format(counters.iloc[-1, [1, 3, 4, 5, 6, 7]].sum()))
    else:
        for i in range(n_states):
            print("{}: {}".format(status_dict[i], counters[i]))
