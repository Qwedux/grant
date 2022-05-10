import sklearn.model_selection

class Param_container:
    class Param:
        def __init__(self, display_mode, display_text:str, param_name:str, values:list) -> None:
            self.display_text_ = display_text
            self.name_ = param_name
            self.values_ = values
            self.display_mode_ = display_mode
    
    def __init__(self, params:list[tuple]) -> None:
        self.params_ = [Param_container.Param(d_mode, d_text, p_name, val) for d_mode, d_text, p_name, val in params]
    
    def get_tracked_params(self) -> list[Param]:
        '''returns list of parameters that will be displayed
        '''
        tacked_params = [param for param in self.params_ if param.display_mode_ != None]
        return tacked_params
    
    def make_param_grid(self, num_models:int=5) -> list[dict]:
        param_grid = []
        param_dict = {param.name_:param.values_ for param in self.params_}
        for configuration in list(sklearn.model_selection.ParameterGrid(param_dict)):
            param_grid.append(dict())
            for index in range(num_models):
                for param_name, value in configuration.items():
                    if isinstance(param_name, str) and param_name.endswith("__random_state"):
                        param_grid[-1][param_name.format(index)] = [value[index]]
                    else:
                        param_grid[-1][param_name.format(index)] = [value]
        return param_grid