from numba import cuda, float32, int32


STATES = 60
ACTIONS = 2
DEATH_AGE = 10
BENEFITS = 30


wp = float32


@cuda.jit(device=True)
def step(states, offset):
    return rand_draw(states, offset, STATES)


@cuda.jit(device=True)
def rand_draw(states, offset, n=1):
    return round(cuda.random.xoroshiro128p_uniform_float32(states, offset)*n)


@cuda.jit
def McCallModel(old_pher, q_vals,
                gamma,
                new_pher, rho_tot, tot_util,
                rng_states):

    x, y = cuda.grid(2)
    offset = x + y * cuda.gridDim.x

    q_shape = states, actions = STATES, ACTIONS

    visits = cuda.local.array(q_shape, int32)
    history = cuda.local.array((DEATH_AGE, 2), int32)
    rewards = cuda.local.array(DEATH_AGE, wp)

    temp_new_pher = cuda.shared.array(q_shape, wp)
    temp_new_pher[cuda.threadIdx.x, cuda.threadIdx.y] = 0

    temp_rho_tot = cuda.shared.array(q_shape, wp)
    temp_rho_tot[cuda.threadIdx.x, cuda.threadIdx.y] = 0

    cuda.syncthreads()

    s = step(rng_states, offset)
    age = -1
    while s >= 0:
        age += 1

        diff = q_vals[s, 0] - q_vals[s, 1]
        if diff > 0:
            a = 0
        elif diff < 0:
            a = 1
        else:
            a = rand_draw(rng_states, offset, 1)

        if a == 0:
            r = 0.0
            for i in range(DEATH_AGE-age):
                r += pow(gamma, i)*s
            sp = -1
        else:
            r = BENEFITS
            sp = step(rng_states, offset) if age < DEATH_AGE - 1 else -1

        visits[s, a] = 0

        history[age, 0] = s
        history[age, 1] = a
        rewards[age] = r

        s = sp

    for i in range(age, 0, -1):
        rewards[i-1] += (gamma*rewards[i])

    for i in range(age+1):
        s = history[i, 0]
        a = history[i, 1]
        r = rewards[i]

        visits[s, a] += 1
        if visits[s, a] == 1:
            # this would normally accound for the rho parameter, but for
            # testing purposes, we keep it as an implied rho = 1.
            cuda.atomic.add(temp_new_pher, (s, a), r - old_pher[s, a])
            cuda.atomic.add(temp_rho_tot, (s, a), 1)

    loc_idx = (cuda.threadIdx.x, cuda.threadIdx.y)

    # add total utilities
    cuda.atomic.add(tot_util, loc_idx, rewards[0])

    cuda.syncthreads()

    # transfer from shared to output
    cuda.atomic.add(new_pher, loc_idx, temp_new_pher[loc_idx])
    cuda.atomic.add(rho_tot, loc_idx, temp_rho_tot[loc_idx])
