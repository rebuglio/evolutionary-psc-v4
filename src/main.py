from scipy import stats

if __name__ == '__main__':
    pass

data = [6,500]
mean, var, std = stats.bayes_mvs(data, alpha=0.95)
print(mean)