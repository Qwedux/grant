from parameter_container import Param_container

def print_table(tabulka):
    tabulka.sort()
    s = [[str(e) for e in row] for row in [tabulka[i] for i in range(min(20, len(tabulka)))]]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '  '.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def display_best_params(param_container:Param_container, search_results:dict):
    vypis = []
    for i in range(len(search_results['rank_test_score'])):
        tracked_params = param_container.get_tracked_params()
        vypis.append([
            "Rank:", search_results['rank_test_score'][i],
            "Cross-val:", "{0:.5f}".format(search_results['mean_test_score'][i]),
            "Mean_time:", "{0:.3f}".format(search_results['mean_fit_time'][i]+search_results['mean_score_time'][i])
        ])
        for tracked_param in tracked_params:
            if tracked_param.display_mode_ == 'chosen_param':
                vypis[-1] += [tracked_param.display_text_, search_results['param_'+tracked_param.name_.format(0)][i]]
            elif tracked_param.display_mode_ == 'all_pos_vals':
                vypis[-1] += [tracked_param.display_text_, *tracked_param.values_]
    print_table(vypis)