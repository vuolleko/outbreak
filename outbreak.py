import numpy as np
import pandas as pd
import scipy.stats as ss


status_dict = {i: v for i, v in enumerate(
    ['latent', 'symptoms-non-infectious', 'latent-infectious', 'symptoms',
     'recovering', 'dying', 'recovered', 'dead'])}
infect_time = 1. / 0.34
pd.set_option('display.width', 1000)


class Infectee:
    """Infected individual, instantiated upon infection."""

    def __init__(self, infector, infection_time):
        self.infector = infector  # individual who infected self
        self.infected = []  # list of individuals infected by self
        self.infection_time = infection_time
        self._last_infection = infection_time  # counter starts

        # set future evolution steps of the infection
        self.status_trajectory = [0]  # ref. `status_dict`
        self.end_times = np.full(len(status_dict), np.nan)  # unit: 1 day
        latent_period = ss.gamma.rvs(2., scale=5.)
        incubation_factor = ss.uniform.rvs(0.8, 0.4)

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
        infectious_period = ss.gamma.rvs(1., scale=5.)
        two_periods = latent_period + infectious_period
        self.end_times[3] = two_periods

        will_recover = ss.bernoulli.rvs(0.3)
        if will_recover:
            self.status_trajectory.extend([4, 6])
            recover_period = ss.gamma.rvs(4., scale=3.)
            time_end = two_periods + recover_period
        else:
            self.status_trajectory.extend([5, 7])
            dying_period = ss.gamma.rvs(4./9., scale=9.)
            time_end = two_periods + dying_period
        self.end_times[self.status_trajectory[-2]] = time_end
        self.end_times += infection_time

        self._status_iter = iter(self.status_trajectory)
        self._status = next(self._status_iter)

    def infect(self, other):
        """Mark `other` as infected by self."""
        self.infected.append(other)
        return other

    @property
    def n_infected(self):
        return len(self.infected)

    @property
    def can_infect(self):
        return (self._status == 2) or (self._status == 3)

    @property
    def status(self):
        return status_dict[self._status]

    @property
    def _time_next(self):
        return self.end_times[self._status]

    def update(self, time):
        """When time comes, update status of infection and infect others."""
        if time >= self._time_next:
            self._status = next(self._status_iter)

        if self.can_infect:
            if time - self._last_infection > infect_time:
                new_infectee = self.infect(Infectee(self, time))
                self._last_infection = time

                return new_infectee
        return None

    def __str__(self):
        str = "Individual {} has infected {} others: {}".format(
            hex(id(self)), self.n_infected, [hex(id(i)) for i in self.infected])
        return str


def simulate():
    time = 0
    max_times = 150
    infected = [Infectee(None, time)]  # start with 1
    counters = pd.DataFrame(index=np.arange(1, max_times),
                            columns=status_dict.values())

    while time < max_times:
        time += 1
        counters.loc[time] = 0
        for i in infected:
            new_i = i.update(time)
            if new_i:
                infected.append(new_i)
            counters.loc[time, i.status] += 1

        if time % 10 == 0:
            print(counters.loc[time:time])

    return counters


if __name__ == '__main__':
    counters = simulate()
    # print(counters)
    print("\n")
    print(counters.iloc[-1])
    print("\nCases reported: {}".format(counters.iloc[-1, [1, 3, 4, 5, 6, 7]].sum()))
