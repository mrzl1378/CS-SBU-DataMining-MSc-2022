def setPlotly():
    import matplotlib.style
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(mpl.rcParamsDefault)


    plt.style.use('seaborn')
    mpl.rcParams['figure.figsize'] = [8, 4]
    mpl.rcParams['figure.dpi'] = 120
    mpl.rcParams['savefig.dpi'] = 300

    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6

    mpl.rcParams['font.size'] = 8
    mpl.rcParams['legend.fontsize'] = 'medium'
    mpl.rcParams['figure.titlesize'] = 'medium'
    mpl.rcParams['axes.titlesize'] = 'medium'
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
                                                color=['#636EFA',
                                                '#EF553B',
                                                '#00CC96',
                                                '#AB63FA',
                                                '#FFA15A',
                                                '#19D3F3',
                                                '#FF6692',
                                                '#B6E880',
                                                '#FF97FF',
                                                '#FECB52']) 

    mpl.rcParams['figure.subplot.left'] = 0.1
    mpl.rcParams['figure.subplot.bottom'] = 0.1
    mpl.rcParams['figure.subplot.right'] = 0.9
    mpl.rcParams['figure.subplot.top'] = 0.9
    mpl.rcParams['figure.subplot.wspace'] = 0.4
    mpl.rcParams['figure.subplot.hspace'] = 0.9


    # mpl.rcParams['grid.alpha'] = 1
    # mpl.rcParams['grid.color'] = 'w'
    # mpl.rcParams['grid.linestyle'] = '-'
    # mpl.rcParams['grid.linewidth'] = 0.5

    mpl.rcParams['image.cmap'] = 'cool'
                                                
