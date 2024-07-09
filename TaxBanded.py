# program to impose wealth tax on unequal society
# stops when set number of years has elapsed
# uses multiple increments per transaction to increase speed
# parameters to set: main program population files, tax rate and reach.

print("running")
import sys
import numpy as np
from numpy import savetxt
from numpy import loadtxt
import random
import time
from sklearn.metrics import mean_absolute_percentage_error
from numba import njit, prange
import matplotlib.pyplot as plt

def allgraphs(populationsize, a, lowerband, midband, topband):
    plt.clf()
    print("in main graph routine")
    plt.subplot(2, 2, 1)
    x = np.asarray(np.arange(1, populationsize + 1), dtype = np.float64) # index number of each agent
    sorted_data = np.sort(a) # ascending
    sorted_wealth = sorted_data[::-1] # descending
    plt.scatter(x ,sorted_wealth, color='red', marker = 'o', s = 1)
    plt.title('Wealth distribution')
    plt.xlabel('log rank')
    plt.ylabel('log wealth')
    plt.yscale('log')
    plt.xscale('log')
    #t = populationsize * topband
    plt.axvline(x = lowerband)
    plt.axvline(x = midband, ls = 'dotted')
    plt.axvline(topband, ls = 'dashed')

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
    plt.axvline(lowerband)
    plt.axvline(topband, ls = 'dashed')

    plt.subplot(2,2,3)
    plt.scatter(richestrank, richestsorted_data, color='red', marker = 'o', s = 1)
    plt.title('top 10%')
    plt.xlabel('log rank')
    plt.ylabel('log wealth')
    fit = np.polyfit(np.log(richestrank), np.log(richestsorted_data), 1)
    yfit = np.exp(np.polyval(fit, np.log(richestrank)))
    error = mean_absolute_percentage_error(richestsorted_data, yfit)
    print("Gradient upper 10%:", fit)
    print("MAPE upper 10%: ", error)
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(richestrank, yfit, color = 'blue')
    plt.axvline(lowerband)
    plt.axvline(x = midband, ls = 'dotted')    
    plt.axvline(topband, ls = 'dashed')

    plt.subplot(2,2,4)
    poorestsorted_data = sorted_wealth[top10percent::] # ascending
    poorestrank = x[top10percent::] # descending
    plt.scatter(poorestrank, poorestsorted_data, color='red', marker = 'o', s = 1)
    plt.title('bottom 90%')
    plt.xlabel('rank')
    plt.ylabel('log wealth')
    plt.axvline(lowerband)
    plt.axvline(midband, ls = 'dotted')
    plt.axvline(topband, ls = 'dashed')
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

def shortgraphs(populationsize, a, lowerband, midband, topband):
    print("in short graphs")
    sorted_data = np.sort(a) # ascending
    sorted_wealth = sorted_data[::-1] # descending
    top10percent = int(len(sorted_wealth) / 10)
    if top10percent > 100000: top10percent = 100000 # limit number of points going to matplotlib
    richestsorted_data = sorted_wealth[0:top10percent] # ascending
    x = np.asarray(np.arange(1, populationsize + 1), dtype = np.float64) # index number of each agent
    richestrank = x[0:top10percent] # descending
    plt.plot(richestrank ,richestsorted_data, color='red', marker = '.', markersize = 1)
    plt.title('top 10%')
    plt.xlabel('log rank')
    plt.ylabel('log wealth')
    fit = np.polyfit(np.log(richestrank), np.log(richestsorted_data), 1)
    yfit = np.exp(np.polyval(fit, np.log(richestrank)))
    error = mean_absolute_percentage_error(richestsorted_data, yfit)
    print("Gradient upper 10%:", fit)
    print("MAPE upper 10%: ", error)
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(lowerband)
    plt.axvline(midband, ls = 'dotted')
    plt.axvline(topband, ls = 'dashed')
    plt.plot(richestrank, yfit, color = 'blue')
    plt.show()

def graphs(populationsize, a, lowerband, midband, topband):
    print("plotting log/log of whole population")
    sorted_data = np.sort(a) # ascending
    sorted_wealth = sorted_data[::-1] # descending
    x = np.asarray(np.arange(1, populationsize + 1), dtype = np.float64) # index number of each agent
    plt.plot(x ,sorted_wealth, color='red', marker = '.', markersize = 1)
    plt.title('Wealth distribution')
    plt.xlabel('log rank')
    plt.ylabel('log wealth')
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(lowerband)
    plt.axvline(midband, ls = 'dotted')
    plt.axvline(topband, ls = 'dashed')
    plt.show()

@njit (parallel = True)
def gini_coefficient(a):
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
    for i in prange(1, n):    
        g = g + ((pxi[i] * pci[i - 1]) - (pci[i] * pxi[i - 1]))
    return g

@njit #(parallel = True)
def progress(a): # graph of wealth as program runs
    a = np.asarray(a)
    g = gini_coefficient(a)
    print("Gini coefficient:", g)

    totalwealth = np.sum(a)
    print("Total wealth:", totalwealth)
    sorted_data = np.sort(a) # ascending
    sorted_wealth = sorted_data[::-1] # descending

    top10percent = int(len(sorted_wealth) / 10)
    bottom90percent = len(sorted_wealth) - top10percent

    richestsorted_data = sorted_wealth[0:top10percent] # ascending
    x = np.asarray(np.arange(1, populationsize + 1), dtype = np.float64) # index number of each agent
    richestrank = x[0:top10percent] # descending

@njit #(parallel = True)
def taketax(a, year, wtax, wthreshold, mtax, mthreshold, ntax, nthreshold, redistribute, taxfreeallowance): # apply wealth tax to population, threshold is set in this function because it varies as the program runs
    print("taking tax")
    sorted_wealth = np.sort(a)[::-1] # sort agents into descending order of wealth
    print("Current year is:", year)
    totalwealth = np.sum(sorted_wealth) # record combined wealth for calculation of amount of tax generated

    # calculate taxes for lower tax band
    if taxfreeallowance == True:
        wideliability = (sorted_wealth[sorted_wealth > wthreshold] - wthreshold) * wtax # calculate tax for those in lower tax band: only tax above threshold
    else:
        wideliability = (sorted_wealth[sorted_wealth > wthreshold]) * wtax # calculate tax for those in lower tax band: all wealth taxed
    print("wide liability:", int(np.sum(wideliability)))
    print("wide liability agent count:", len(wideliability))

    if taxfreeallowance == True:
        midliability = (sorted_wealth[sorted_wealth > mthreshold] - mthreshold) * (mtax - wtax) # calculate tax for those in middle tax band: only tax above threshold
    else:
        midliability = (sorted_wealth[sorted_wealth > mthreshold]) * (mtax - wtax) # calculate tax for those in middle tax band: all wealth taxed   
    print("middle liability:", int(np.sum(midliability)))
    print("middle liability agent count:", len(midliability))

    # calculate taxes for upper tax band
    if taxfreeallowance == True:
        narrowliability = (sorted_wealth[sorted_wealth > nthreshold] - nthreshold) * (ntax - mtax) # identify those in upper tax band and apply allowance: only tax above allowance
    else:
        narrowliability = (sorted_wealth[sorted_wealth > nthreshold]) * (ntax - mtax) # identify those in upper tax band and apply allowance: all wealth taxed
    print("narrow liability:", int(np.sum(narrowliability)))
    print("narrow liability agent count:", len(narrowliability))
#    print("lowest liability of upper band:", min(narrowliability))

    # apply taxes
    sorted_wealth[sorted_wealth > wthreshold] -= wideliability # update wealth by taxing lower band
    sorted_wealth[sorted_wealth > mthreshold] -= midliability # take tax for middle band
    sorted_wealth[sorted_wealth > nthreshold] -= narrowliability # subtract tax for higher band

    currentwealth = np.sum(sorted_wealth) # calculate new combined wealth
    taxtake = totalwealth - currentwealth # calculate total tax revinue
    print("tax take:", int(taxtake))
    if redistribute == True:
        print("redistributing")
        repayment = taxtake / len(sorted_wealth) # calculate average tax payment
        sorted_wealth += repayment # distribute tax take equally to all agents
    return sorted_wealth

def setwealthpower(a):
    p = len(a)
    if p == 50000000:
        wealthpower = 1.09 # population 50,000,000
    elif p == 40000000:
        wealthpower = 1.09 # population 40,000,000
    elif p == 30000000:
        wealthpower = 1.09 # population 30,000,000
    elif p == 20000000:
        wealthpower = 1.09 # population 20,000,000
    elif p == 10000000:
        wealthpower = 1.05 # population 10,000,000
    elif p == 1000000:
        wealthpower = 1.11 # population 1,000,000
    elif p == 100000:
        wealthpower = 1.14 # population 100,000
    else:
        print("wealthpower not set for this population: using 1.1")
        wealthpower = 1.1
    return wealthpower

def run_simulation(pop, inc, populationsize, targetyears, widewealthtaxrate, widewealthtaxreachreach, midwealthtaxrate, midwealthtaxreach, narrowwealthtaxrate,
				 narrowwealthtaxreach, update, redistribute, wealthratiohistory, taxfreeallowance):
    print("running simulation")
    agentwealth = pop # initialise wealth of all agents
    wealthpower = setwealthpower(agentwealth) # set exponent for calculating probability of gaining an increment during each transaction
    maxwealth = max(agentwealth)
    maxwealthatstart = maxwealth
    wealthratio = np.max(agentwealth) / np.median(agentwealth)
    increment = inc # amount awarded in each transaction
    proportiontoincrement = 0.01 # increment part of population in each transaction (0.0 means single agent mode)
    numbertoincrement = int(populationsize * proportiontoincrement)
    print(numbertoincrement)
    growthrate = 0.02588 # lambda coefficient from Vallejos et al.
    averagepercentageincrease = 0.0 # initialize average annual increase in wealth of richest agent

    tax = False # dont do tax calculations if wealth tax rate or reach is zero
    if ((widewealthtaxrate > 0.0 and widewealthtaxreach > 0.0) or (midwealthtaxrate > 0.0 and midwealthtaxreach > 0.0) or (narrowwealthtaxrate > 0.0 and narrowwealthtaxreach > 0.0)):
        tax = True

    initialtotalwealth = np.asarray(np.sum(agentwealth), dtype = np.float64) # total wealth at the start
    targetwealth = initialtotalwealth + (growthrate * initialtotalwealth) # wealth after 1 year of growth
    year = 0
    oldmax = np.asarray(max(agentwealth), dtype=np.float64) # store the current wealth of the richest agent
    print("entering while loop")
    stop = False
    icount = 0 # count of increments
    ivalue = 0 # combined value of increments
    while stop == False: # add 1 or more increment to selected members of the population
        powerarray = np.asarray(np.power(agentwealth, wealthpower)) # calculate ability to attract new wealth for each individual
        totalwealthpower = np.sum(powerarray)
        probabilityarray = powerarray / totalwealthpower # probability of obtaining next increment for each agent - eqn. 6 in Vallejos et al.
        currentagent = 0
        population = np.cumsum(probabilityarray) # calculate cumulative probability of each agent obtaining increment

        if proportiontoincrement > 0.0: # multiple alocation
            currentagent = np.searchsorted((population), np.random.random(numbertoincrement)) # select the agents that each get an increment
            for i in currentagent:
                agentwealth[i] += increment # award increment
                icount += 1
                ivalue += increment
        else: # single allocation
            currentagent = np.searchsorted((population), np.random.random()) # choose single agent to get increment
            agentwealth[currentagent] += increment

        currentwealth = np.sum(agentwealth) # calculate new total wealth
        if currentwealth >= targetwealth: # a year of growth has occurred
            year += 1
            print("\n", year, "simulated years elapsed")
            widenumbertotax = midnumbertotax = narrownumbertotax = 0
            if tax == True: # calculate and collect wealth tax
                print("Update status:", update)
                sorted_wealth = np.sort(agentwealth)[::-1] # sort agents into descending order of wealth
                if year == 1 or (year > 1 and update == True):
                    # calculate taxes for lower tax band
                    widenumbertotax = int(widewealthtaxreach * len(sorted_wealth)) # number of people in lower tax band
                    widethreshold = sorted_wealth[widenumbertotax] # set threshold for lower tax band ( = wealth of poorest in band)
                    # calculate taxes for medium tax band
                    midnumbertotax = int(midwealthtaxreach * len(sorted_wealth)) # number of people in middle tax band
                    midthreshold = sorted_wealth[midnumbertotax] # set threshold for middle tax band ( = wealth of poorest in band)
                    # calculate taxes for upper tax band
                    narrownumbertotax = int(narrowwealthtaxreach * len(sorted_wealth)) # number of people in higher tax band
                    narrowthreshold = sorted_wealth[narrownumbertotax] # set upper band ( = wealth of poorest in band)
                if update == False:
                    widenumbertotax = len(sorted_wealth[sorted_wealth > widethreshold])
                    midnumbertotax = len(sorted_wealth[sorted_wealth > midthreshold])
                    narrownumbertotax = len(sorted_wealth[sorted_wealth > narrowthreshold])
                    
                print("number in lower tax band:", widenumbertotax) # lower tax band
                print("lower threshold", widethreshold)
                print("number in middle tax band:", midnumbertotax) # lower tax band
                print("middle threshold", midthreshold)
                print("number in upper tax band:", narrownumbertotax) # upper tax band
                print("upper threshold:", narrowthreshold)

                agentwealth = taketax(sorted_wealth, year, widewealthtaxrate, widethreshold, midwealthtaxrate, midthreshold,
                                    narrowwealthtaxrate, narrowthreshold, redistribute, taxfreeallowance)

            ind = np.argmax(agentwealth) # locate richest agent
            newmaxwealth = np.asarray(np.max(agentwealth), dtype=np.float64) # calculate updated wealth of richest agent after tax
            wealthratio = newmaxwealth / np.median(agentwealth) # calculate wealth ratio
            print("wealth ratio was", int(wealthratio))
            print("old max wealth", int(oldmax), "new max wealth", int(newmaxwealth))
            maxincome = ((newmaxwealth - oldmax) / oldmax) * 100 # calculate percentage rate of return after tax
            print("richest gained", maxincome, "percent, after tax")
            averagepercentageincrease += maxincome # add up each annual percentage increase in wealth of richest - used for calibrating wealthpower coefficient
            print("increment", int(increment))
            print("earnings by addition", int(newmaxwealth - oldmax))
            oldmax = newmaxwealth # reset starting wealth of richest
            wealthratiohistory[year] = wealthratio

            if year >= targetyears: # check for simulation stopping condition that wes set in the main program
                stop = True
                print("stop")
                averagepercentageincrease = averagepercentageincrease / year # calculate overall rate of return for richest
                print("average rate of return of richest:", averagepercentageincrease, "percent")
                break
            targetwealth = targetwealth + (growthrate * currentwealth) # set new stopping condition for following year
            increment = increment + (increment * growthrate) # set new increment
            print("new increment:", int(increment))
            wealthratio = np.max(agentwealth) / np.median(agentwealth)
            print("Current wealth ratio:", round(wealthratio))
            setwealthpower(agentwealth) # update beta coefficient for next year
    print("returning from main loop")
    print("increment count:", icount)
    print("increment value:", ivalue)
    return agentwealth, wealthratiohistory, widenumbertotax, midnumbertotax, narrownumbertotax

def finalresults(a): # analyze results at the end of the simulation
    print("\nProcessing complete")
    a = np.asarray(a)
    x = np.asarray(np.arange(1, populationsize + 1), dtype = np.float64) # index number of each agent
    g = gini_coefficient(a)
    print("\nGini coefficient:", g)

    totalwealth = np.sum(a)
    print("Total wealth:", totalwealth)
    sorted_data = np.sort(a) # ascending
    sorted_wealth = sorted_data[::-1] # descending

    top10percent = int(len(sorted_wealth) / 10)
    bottom90percent = len(sorted_wealth) - top10percent

    richestsorted_data = sorted_wealth[0:top10percent] # ascending
    x = np.asarray(np.arange(1, populationsize + 1), dtype = np.float64) # index number of each agent
    richestrank = x[0:top10percent] # descending

    fit = np.polyfit(np.log(richestrank), np.log(richestsorted_data), 1)
    yfit = np.exp(np.polyval(fit, np.log(richestrank)))
    error = mean_absolute_percentage_error(richestsorted_data, yfit)
    print("Gradient upper 10%:", fit)
    print("MSE upper 10%: ", error)

    poorestsorted_data = sorted_wealth[top10percent::] # ascending
    poorestrank = x[top10percent::] # descending

    fit = np.polyfit(np.log(poorestrank), np.log(poorestsorted_data), 1)
    yfit = np.exp(np.polyval(fit, np.log(poorestrank)))
    error = mean_absolute_percentage_error(np.log(poorestsorted_data), yfit)

# main program
start_time = time.time()
targetyears = 100
wealthratiohistory = np.zeros(targetyears + 1) # create an array to store wealth ratio each year
widewealthtaxrate = 0.0 # wealth tax rate: wide band (0.01)
widewealthtaxreach = 0.0 # proportion of population liable to tax: wide band
midwealthtaxrate = 0.0 # wealth tax rate for middle band
midwealthtaxreach = 0.0 # proportion of population liable to middle band
narrowwealthtaxrate = 0.3 # wealth tax rate: narrow band
narrowwealthtaxreach = 0.01 # proportion of population liable to tax: narrow band
updatethresholds = False # update tax thresholds every year
redistribute = True # whether to resistribute tax taken or store it
taxfreeallowance = True # whether to ignore wealth below tax threshold
try:
    inputpopulation = loadtxt('pop10000000gini8distribution.csv') # load saved population
    agentwealth = np.asarray(inputpopulation, dtype = np.float64)
    del inputpopulation # make sure memory is freed up immediately
    print("loaded population of", len(agentwealth))
except IOError:
    print("file not found")
    sys.exit()

try:
    f = open("pop10000000gini8increment.txt", "r") # get saved increment
    increment = float(f.readline())
    f.close()
except IOError:
    print("file not found")
    sys.exit()

populationsize = len(agentwealth)
initmax = np.max(agentwealth)
initmedian = np.median(agentwealth)
initmin = np.min(agentwealth)
initgini = gini_coefficient(agentwealth) # save starting gini
initratio = np.max(agentwealth) / initmedian # save starting wealth ratio
populationa = np.sort(agentwealth)
populationd = populationa[::-1]
del populationa
n1 = int(len(populationd) * 0.01)
n10 = int(len(populationd) * 0.1)
totalwealth = np.sum(populationd)
wealthtop1pc = np.sum(populationd[0:n1])
wealthtop10pc = np.sum(populationd[0:n10])
del populationd
initsharetop1pc = (wealthtop1pc / totalwealth) * 100
initsharetop10pc = (wealthtop10pc / totalwealth) * 100
wealthratiohistory[0] = np.max(agentwealth) / np.median(agentwealth)
print("Initial Gini coefficient:", round(initgini, 3))
print("Wealth ratio at start:", round(initratio, 3))
print("preparing to run_simulation")
agentwealth, wealthratiohistory, widenumbertotax, midnumbertotax, narrownumbertotax = run_simulation(agentwealth, increment, populationsize, targetyears, 
		widewealthtaxrate, widewealthtaxreach, midwealthtaxrate, midwealthtaxreach, narrowwealthtaxrate, narrowwealthtaxreach,
		updatethresholds, redistribute, wealthratiohistory, taxfreeallowance)
print("\n")
print("Results")
print("Initial maximum wealth:", (f"{int(initmax):,}"))
print("Final maximum wealth:", (f"{int(np.max(agentwealth)):,}"))
print("Initial median wealth:", (f"{int(initmedian):,}"))
print("Final median wealth:", (f"{int(np.median(agentwealth)):,}"))
print("Initial minimum wealth:", (f"{int(initmin):,}"))
print("Final minimum wealth:", (f"{int(np.min(agentwealth)):,}"))
populationa = np.sort(agentwealth) # ascending order
populationd = populationa[::-1] # descending order
del populationa
n1 = int(len(populationd) * 0.01) # top 1%
n10 = int(len(populationd) * 0.1) # top 10%
coefficient = n10 * (1 / (np.sum(np.log(populationd[0:n10] / np.min(populationd[0:n10])))))
print("Pareto Coefficient: ", coefficient)
totalwealth = np.sum(populationd)
wealthtop1pc = np.sum(populationd[0:n1])
wealthtop10pc = np.sum(populationd[0:n10])
del populationd
finalsharetop1pc = (wealthtop1pc / totalwealth) * 100
finalsharetop10pc = (wealthtop10pc / totalwealth) * 100
print("Initial share top 1%:", int(initsharetop1pc))
print("Final share top 1%:", int(finalsharetop1pc))
print("Initial share top 10%:", int(initsharetop10pc))
print("Final share top 10%:", int(finalsharetop10pc))
print("Initial Gini coefficient:", round(initgini, 3))
print("Final Gini coefficient:", round(gini_coefficient(agentwealth), 3))
print("Wealth ratio at start:", (f"{int(initratio):,}"))
finalratio = np.max(agentwealth) / np.median(agentwealth)
print("Wealth ratio at end:", (f"{int(finalratio):,}"))
print("Percentage change in ratio:", int(((finalratio - initratio) / initratio) * 100))
print("\nProcessing took: ", round((time.time() - start_time)/60, 1)," Minutes")

#finalresults(agentwealth)
g = input("Show graphs? y/n ")
if g != 'n':
    x = np.asarray(np.arange(0, targetyears + 1), dtype = int) # index number of each agent
    plt.plot(x ,wealthratiohistory, color='red', linewidth = 1)
    plt.title('Wealth ratio')
    plt.xlabel('year')
    plt.ylabel('wealth ratio')
    plt.show()
    graphs(populationsize, agentwealth, widenumbertotax, midnumbertotax, narrownumbertotax)
    if populationsize >= 10000000:
        shortgraphs(populationsize, agentwealth, widenumbertotax, midnumbertotax, narrownumbertotax)
    else:
        allgraphs(populationsize, agentwealth, widenumbertotax, midnumbertotax, narrownumbertotax)
#run_simulation.parallel_diagnostics(level=3)
