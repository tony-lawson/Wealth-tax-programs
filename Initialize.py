# program to generate unequal society and save it for later analysis using VallejosTax.py and variants
# stops whwn a target gini coefficient is reached
# parameters to set include targetgini, populationsize, growthrate, proportiontoincrement, and wealthpower list
# allows multiple increments per transaction to increase speed

print("running")
import numpy as np
from numpy import savetxt
import random
import time
from sklearn.metrics import mean_absolute_percentage_error
from numba import njit
import matplotlib.pyplot as plt
from os.path import exists

def graphs(a):
    plt.clf()
    populationsize = len(a)
    x = np.asarray(np.arange(1, populationsize + 1), dtype = np.float64) # index number of each agent
    plt.subplot(2, 2, 1)
    plt.scatter(x, a, color='red', s = 1)
    plt.title('all agents')
    plt.xlabel('Agent index')
    plt.ylabel('Wealth')
    
    sorted_data = np.sort(a) # ascending
    sorted_wealth = sorted_data[::-1] # descending
    plt.subplot(2, 2, 2)
    plt.scatter(x, sorted_wealth, color='red', s = 1)
    plt.title('all agents')
    plt.xlabel('Agent index')
    plt.ylabel('Wealth')

    totalwealth = np.sum(a)
    print("Total wealth:", totalwealth)

    top10percent = int(len(sorted_wealth) / 10)
    bottom90percent = len(sorted_wealth) - top10percent

    richestsorted_data = sorted_wealth[0:top10percent] # ascending
    richestrank = x[0:top10percent] # descending
   
    plt.subplot(2,2,3)
    fit = np.polyfit(np.log(richestrank), np.log(richestsorted_data), 1)
    yfit = np.exp(np.polyval(fit, np.log(richestrank)))
    error = mean_absolute_percentage_error(richestsorted_data, yfit)
    print("Gradient upper 10%:", fit)
    print("MAPE upper 10%: ", error)
    print("target richest:", yfit[0])
    print("actual richest:", richestsorted_data[0])
    toperror = (((richestsorted_data[0] - yfit[0]) / yfit[0]) * 100)
    #richestsorted_data = yfit # snap modelled data to trend line
    print(toperror)    
    plt.scatter(richestrank ,richestsorted_data, color='red', marker = 'o', s = 4)
    plt.title('top 10%')
    plt.xlabel('log rank')
    plt.ylabel('log wealth')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(richestrank, yfit, color = 'blue')

    plt.subplot(2,2,4)
    poorestsorted_data = sorted_wealth[top10percent::] # ascending
    poorestrank = x[top10percent::] # descending

    plt.scatter(poorestrank, poorestsorted_data, color='red', marker = 'o', s = 4)
    plt.title('bottom 90%')
    plt.xlabel('rank')
    plt.ylabel('log wealth')
    fit = np.polyfit(np.log(poorestrank), np.log(poorestsorted_data), 1)
    yfit = np.exp(np.polyval(fit, np.log(poorestrank)))
    error = mean_absolute_percentage_error(np.log(poorestsorted_data), yfit)
    #print("Gradient lower 90%:", fit)
    #print("MAPE lower 90%: ", error)
    plt.yscale('log')
    #plt.xscale('log')    
    plt.plot(poorestrank, yfit, color = 'blue')
    plt.tight_layout()
    plt.show()
    #plt.draw()
    #plt.pause(0.01)

def shortgraphs(a):
    print("in short graphs")
    sorted_data = np.sort(a) # ascending
    sorted_wealth = sorted_data[::-1] # descending    
    top10percent = int(len(sorted_wealth) / 10)
    if top10percent > 100000: top10percent = 100000 # limit number of points going to matplotlib
    richestsorted_data = sorted_wealth[0:top10percent] # ascending
    populationsize = len(a)
    x = np.asarray(np.arange(1, populationsize + 1), dtype = np.float64) # index number of each agent
    richestrank = x[0:top10percent] # descending
   
    plt.scatter(richestrank ,richestsorted_data, color='red', marker = 'o', s = 4)
    plt.title('top 10%')
    plt.xlabel('log rank')
    plt.ylabel('log wealth')
    fit = np.polyfit(np.log(richestrank), np.log(richestsorted_data), 1)
    yfit = np.exp(np.polyval(fit, np.log(richestrank)))
    error = mean_absolute_percentage_error(richestsorted_data, yfit)
    print("Gradient upper 10%:", fit)
    print("MAPE upper 10%: ", error)
    print("target richest:", yfit[0])
    print("actual richest:", richestsorted_data[0])
    toperror = (((richestsorted_data[0] - yfit[0]) / yfit[0]) * 100)
    #richestsorted_data = yfit # snap modelled data to trend line
    print(toperror)
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(richestrank, yfit, color = 'blue')
    plt.show()
   
@njit#(parallel = True)
def gini_coefficient(a):
    """Compute Gini coefficient of array of values"""
    x = np.asarray(a)
    w = np.ones_like(x)
    n = x.size
    wxsum = np.sum(w * x)
    wsum = np.sum(w)
    sxw = np.argsort(x)
    sx = x[sxw] * w[sxw]
    sw = w[sxw]
    pxi = np.cumsum(sx) / wxsum
    pci = np.cumsum(sw) / wsum
    g = 0.0
    for i in np.arange(1, n):
        g = g + pxi[i] * pci[i - 1] - pci[i] * pxi[i - 1]
    return g

@njit
def setwealthpowersmall(a): # population 100,000
    g = gini_coefficient(a)
    print("Gini:", g)
    wealthpower = 1.1
    return wealthpower

@njit
def setwealthpowermedium(a): # population 1,000,000
    g = gini_coefficient(a)
    print("Gini:", g)
    print("in medium")
    wealthpower = 1.1
    return wealthpower

@njit
def setwealthpowerlarge(a): # population 10,000,000
    g = gini_coefficient(a)
    print("Gini:", g)
    wealthpower = 1.2
    return wealthpower

@njit#(parallel = True)
def mainloop():
    wealthpower = 1.2 #1.36 # beta # initial value
    increment = 0.1 #0.01 # delta omega
    proportionunique = 0.1 # frequency of limiting agents selected for increment to avoid runaway gains
    agentwealth = np.asarray([1.0] * populationsize, dtype = np.float64) # initialise wealth of all agents to 1

    if populationsize >= 5000000:
        wealthpower = setwealthpowerlarge(agentwealth) # set initial beta coefficient for 10,000,000 population
        proportionunique = 0.1
    elif populationsize > 500000 and populationsize < 5000000:
        wealthpower = setwealthpowermedium(agentwealth) # set initial beta for 1,000,000 population
        proportionunique = 0.0
    else:
        wealthpower = setwealthpowersmall(agentwealth) # set initial beta coefficient for 100,000 population
        proportionunique = 0.0

    initialwealth = np.asarray(np.sum(agentwealth), dtype = np.float64) # total wealth at the start
    targetwealth = initialwealth + (growthrate * initialwealth) # wealth after 1 year of growth
    # xindex = np.asarray(np.arange(1, populationsize + 1), dtype = np.float64) # index number of each agent
    powerarray = np.asarray(np.power(agentwealth, wealthpower)) # calculate wealth for each individual
    totalwealthpower = np.sum(powerarray)
    probabilityarray = powerarray / totalwealthpower
    currentagent = 0
    year = 0
    stop = False
    # maxwealth = max(agentwealth)

    while stop == False:
        population = np.cumsum(probabilityarray)
        if proportiontoincrement > 0.0:
            #for i in range(populationsize * proportiontoincrement): # do several increments for each transaction
            #print("in multiple increment")
            #print(population, proportiontoincrement)
            n = int(len(population) * proportiontoincrement)
            currentagents = np.searchsorted((population), np.random.random(size = n))
            #print(currentagents)
            r = np.random.random()
            if r < proportionunique: currentagents = np.unique(currentagents)
            for a in currentagents:
                agentwealth[a] += increment
            #print("updating selection probabilities in multi increment mode")
            powerarray = np.asarray(np.power(agentwealth, wealthpower)) # calculate wealth for each individual
            totalwealthpower = np.sum(powerarray)
            probabilityarray = powerarray / totalwealthpower
            #print(transaction, currentagents)
        else:
            #print ("in single increment")
            currentagent = np.searchsorted((population), np.random.random())
            agentwealth[currentagent] += increment
            #print("updating selection probabilities in single increment mode")
            powerarray = np.asarray(np.power(agentwealth, wealthpower)) # calculate wealth for each individual
            totalwealthpower = np.sum(powerarray)
            probabilityarray = powerarray / totalwealthpower

        currentwealth = np.sum(agentwealth)
        if currentwealth >= targetwealth: # a year of growth has occurred
            year += 1
            print(year, "simulated years elapsed")
            # wealthincrease = ((max(agentwealth) - maxwealth) / maxwealth) * 100
            g = gini_coefficient(agentwealth)
            populationa = np.sort(agentwealth)
            populationd = populationa[::-1]
            n1 = int(len(populationd) * 0.01)
            n10 = int(len(populationd) * 0.1)
            totalwealth = np.sum(populationd)
            wealthtop1pc = np.sum(populationd[0:n1])
            wealthtop10pc = np.sum(populationd[0:n10])
            sharetop1pc = (wealthtop1pc / totalwealth) * 100
            sharetop10pc = (wealthtop10pc / totalwealth) * 100
            wealthratio = np.max(populationd) / np.median(populationd)
            print("Wealth ratio:", round(wealthratio))
            print("Top 1%", int(sharetop1pc))
            print("Top 10%", int(sharetop10pc))
            print(" -- Richest:", max(agentwealth))

            if g >= targetgini:
                print("stop")
                stop = True
                break
            targetwealth = targetwealth + (growthrate * currentwealth)
            increment = increment + (increment * growthrate)
            if populationsize >= 5000000:
                wealthpower = setwealthpowerlarge(agentwealth) # update beta coefficient population 5,000,000 and over
            elif populationsize > 500000 and populationsize < 5000000: 
                setwealthpowermedium(agentwealth) # population of 500,000 to 5,000,000
            else:
                setwealthpowersmall(agentwealth) # population under 500,000
    print("returning from main loop")
    print("increment was", increment)
    return agentwealth, increment

# main program
start_time = time.time()
populationsize = 10000000
# numba considers global variables as compile-time constants
targetgini = 0.9      # they cannot be changed in njit called functions
proportiontoincrement = 0.01 # increment part of population in each transaction
growthrate = 0.02588 # lambda

agentwealth, increment = mainloop()

print("\nProcessing took: ", (time.time() - start_time)/60, " Minutes")

g = input("Show graphs? y/n ")
if g != 'n':
    if populationsize > 100000: # size limit in matplotlib
        shortgraphs(agentwealth) # only graph top 10%
    else:
        graphs(agentwealth)

# save wealth distribution and increment to file according to program parameters (0 to 9)
print("saving population wealth distribution")
distributionfile = ("pop" + str(populationsize) + "gini" + str(int(targetgini*10)) + "distribution1.csv")
if exists(distributionfile):

    response = input("File already exists: overwrite y/n ")
    if response == 'y':
        savetxt(distributionfile, agentwealth, delimiter = ',')
        incrementfile = ("pop" + str(populationsize) + "gini" + str(int(targetgini*10)) + "increment1.txt")
        f = open(incrementfile, "w")
        f.write(str(increment))
        f.close()
        print("old file overwritten")
else:
    savetxt(distributionfile, agentwealth, delimiter = ',')
    incrementfile = ("pop" + str(populationsize) + "gini" + str(int(targetgini*10)) + "increment0.txt")
    f = open(incrementfile, "w")
    f.write(str(increment))
    f.close()
    print("results stored")

#mainloop.parallel_diagnostics(level=3)
