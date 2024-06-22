def init_change_mod(change_params, start_landscapes, end_landscapes, start_t=200, end_t=201, n_steps=1):
    # Import the necessary modules
    import geonomics as gnx
    from evodoodle import init_mod

    # Update the params
    change_params = set_landscapes(change_params, start_landscapes, end_landscapes, start_t=start_t, end_t=end_t, n_steps=n_steps)

    # Make our params dict into a proper Geonomics ParamsDict object
    change_params = gnx.make_params_dict(change_params, 'demo')

    # Then use it to make a model
    mod = gnx.make_model(parameters=change_params, verbose=True)

    # Burn in the model for 10000 steps
    mod.walk(T=10000, mode='burn')

    return mod

def set_landscapes(change_params, start_landscapes, end_landscapes, start_t=200, end_t=201, n_steps=1):    
    # Set the landscape dimensions
    change_params['landscape']['main']['dim'] = start_landscapes['population_size'].shape
    
    # Set the start landscape layers
    change_params['landscape']['layers']['population_size']['init']['defined']['rast'] = start_landscapes['population_size']
    change_params['landscape']['layers']['connectivity']['init']['defined']['rast'] = start_landscapes['connectivity']
    change_params['landscape']['layers']['environment']['init']['defined']['rast'] = start_landscapes['environment']
    
    # Set the end landscape layers
    change_params['landscape']['layers']['population_size']['change'][0]['change_rast'] = end_landscapes['population_size']
    change_params['landscape']['layers']['connectivity']['change'][0]['change_rast'] = end_landscapes['connectivity']
    change_params['landscape']['layers']['environment']['change'][0]['change_rast'] = end_landscapes['environment']

    # Set the change parameters
    change_params['landscape']['layers']['population_size']['change'][0]['start_t'] = start_t
    change_params['landscape']['layers']['connectivity']['change'][0]['end_t'] = end_t
    change_params['landscape']['layers']['environment']['change'][0]['n_steps'] = n_steps

    return(change_params)


# Draw landscapes
population_size1 = draw_landscape(d = 15)
connectivity1 = draw_landscape(d = 15)
environment1 = draw_landscape(d = 15)

population_size2 = edit_landscape(population_size1)
connectivity2 = edit_landscape(connectivity1)
environment2 = edit_landscape(environment1)

start_landscapes = {
    'population_size': population_size1,
    'connectivity': connectivity1,
    'environment': environment1
}

end_landscapes = {
    'population_size': population_size2,
    'connectivity': connectivity2,
    'environment': environment2
}

mod = init_change_mod(change_params, start_landscapes, end_landscapes, start_t = 10, end_t=11, n_steps=1)

plot_popgen(mod, lyr_num=(1, 0, 2))
import copy

mod.walk(20)
plot_popgen(mod, lyr_num=(1, 0, 2))

mod300 = copy.deepcopy(mod)

mod.walk(200)
plot_popgen(mod, lyr_num=(1, 0, 2))