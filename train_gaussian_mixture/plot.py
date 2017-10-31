# -*- coding: utf-8 -*-
import sampler, pylab, os
import seaborn as sns
from model import discriminator_params, generator_params, gan
from args import args
sns.set(font_scale=2)
sns.set_style("white")

def plot_kde(data, dir=None, filename="kde", color="Greens"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	bg_color  = sns.color_palette(color, n_colors=256)[0]
	ax = sns.kdeplot(data[:, 0], data[:,1], shade=True, cmap=color, n_levels=30, clip=[[-4, 4]]*2)
	ax.set_axis_bgcolor(bg_color)
	kde = ax.get_figure()
	pylab.xlim(-4, 4)
	pylab.ylim(-4, 4)
	kde.savefig("{}/{}.png".format(dir, filename))

def plot_scatter(data, dir=None, filename="scatter", color="blue"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color=color)
	pylab.xlim(-4, 4)
	pylab.ylim(-4, 4)
	pylab.savefig("{}/{}.png".format(dir, filename))

def plot_heatmap_fast(value, dir='plot', filename="heatmap", prob=True, suffix=''):
    '''
    [input format]
      value[0,0] = value of (x0,y0)
      value[0,1] = value of (x0,y1)
      value[0,2] = value of (x0,y2)
      ...
      value[4,4] = value of (x4,y4)

    [Usage]
      mesh = np.mgrid[-4:4:0.1, -4:4:0.1].transpose(1,2,0).reshape(-1,2)
      predict = sess.run( foo(x) , feed_dict={x:mesh} )
      predict = predict.reshape(80,80).transpose(1,0)
      predict = np.flip(predict, 0)
      plot_heatmap_fast(predict, train_sample_directory, 'Degeneration_{}'.format(t), prob=False)
    '''
    check = np.zeros([])
    assert len(value.shape) == 2
    assert type(value) == type(check)
    if os.path.exists(dir) is False:
        os.mkdir(dir)
    labels = list(np.arange(-4,4).astype(int))
    labellist = [''] * 80
    for i,l in enumerate(labels):
        labellist[i*10] = l
    labellist[-1] = 4

    fig = pylab.gcf()
    fig.set_size_inches(20.0, 16.0)
    pylab.clf()
    if prob == True:
        if value.min() < 0 and value.max() > 1:
            value = 1. / (1+np.exp(-value))
        ax = sns.heatmap(value, cbar=True, xticklabels=labellist, yticklabels=labellist[::-1], cbar_kws={'ticks':[0,.2,.4,.6,.8,1]}, vmax=1, vmin=0)
    else:
        ax = sns.heatmap(value, cbar=True, xticklabels=labellist, yticklabels=labellist[::-1])
    hmap = ax.get_figure()
    pylab.title('{}\n{}'.format(filename, suffix), fontsize=24)
    hmap.savefig(os.path.join(dir, "{}{}.png".format(filename, suffix)))

def main():
	num_samples = 10000
	samples_true = sampler.gaussian_mixture_circle(num_samples, num_cluster=generator_params["config"]["num_mixture"], scale=2, std=0.2)
	plot_scatter(samples_true, args.plot_dir, "scatter_true")
	plot_kde(samples_true, args.plot_dir, "kde_true")
	samples_fake = gan.to_numpy(gan.generate_x(num_samples, test=True))
	plot_scatter(samples_fake, args.plot_dir, "scatter_gen")
	plot_kde(samples_fake, args.plot_dir, "kde_gen")

if __name__ == "__main__":
	main()
