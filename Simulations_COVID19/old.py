 
    def save_table(self, path = '', file = 'phenom.json'):
        data = pd.DataFrame()

        field = 'deaths'
        vmean = []
        vmin = []
        vmax = []
        model = []
        Countries = []
        times = []
        datas = []
        for cn in self.post_pred.keys():
            for Country in self.data.Country.unique():
                mean_value = np.mean(self.post_pred[cn][Country].T,axis=1)   
                std_value = np.std(self.post_pred[cn][Country].T,axis=1)
                vmean += [mean_value]
                vmin += [mean_value + std_value]
                vmax += [mean_value - std_value]
                N = len(vmax[0])
                Countries += [Country for i in range(N)]
                model += [cn for i in range(N)]
                times += date_list(self.data[(self.data['Country'] == Country)].time.head(1).values[0], N)
                datas += [np.pad(self.data[(self.data['Country'] == Country)][field], (0, N-len(self.data[(self.data['Country'] == Country)][field])), 'constant', constant_values=(0, np.nan))]
                
        data['vmean'] = np.concatenate(vmean)
        data['vmin'] = np.concatenate(vmin)
        data['vmax'] = np.concatenate(vmax)
        data['data'] = np.concatenate(datas)
        data['Country'] = Countries
        data['model'] = model
        data['time'] = times
        data.to_json(path+file)