import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df=pd.read_csv("cardata.csv",nrows = 1040,)
df.dropna()
df.drop_duplicates()
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
print(df)

df['class'] = df['class'].replace('unacc', 'unacceptable')
df['class'] = df['class'].replace('acc', 'acceptable')
print(df)

meanfor5moredoors =(5+6)/2
print(meanfor5moredoors)
df['doors'] = df['doors'].replace('5more', meanfor5moredoors)
df['doors'] = df['doors'].astype(float)

meanformoreseater= (5+9)/2
df['persons'] = df['persons'].replace('more', meanformoreseater)
df['persons'] = df['persons'].astype(float)
print(df.head(n=500))

df['type'] = ''
df.loc[df['doors'] == 2.0, 'type'] = 'coupe'
df.loc[df['doors'] == 3.0, 'type'] = 'hatchback'
df.loc[df['doors'] == 4.0, 'type'] = 'sedan'
df.loc[df['doors'] == meanfor5moredoors, 'type'] = 'family hatchback'
print(df.head(n=900))

mean = df['doors'].mean()
median = df['doors'].median()
std_dev = df['doors'].std()
mode = df['doors'].mode().to_string()
print(f'Mean for car doors:{mean:.2f}')
print("Median for car doors:", median)
print(f'Standard Deviation for car doors:{std_dev:.2f}')

modelugboot =df['lug_boot'].mode()[0]
print('Car luggage boots sizes are usually',modelugboot)

# Ensure there are no NaNs or invalid values in the data
df = df.dropna(subset=['doors', 'maint', 'buying', 'class'])

value_counts = df['type'].value_counts()
countcoupe=value_counts.get('coupe', 0)
countsedan=value_counts.get('sedan', 0)
counthatchback=value_counts.get('hatchback', 0)
countfamhatchback=value_counts.get('family hatchback', 0)

font1 = {'family': 'serif', 'color': 'blue', 'size' :20}
font2 = {'family': 'serif', 'color': 'red', 'size' :15}
x = np.array(["coupe", "sedan", "familyhatchback", "hatchback"])
y = np.array([countcoupe,countsedan,countfamhatchback,counthatchback])
plt.title('Number of different types of Car', fontdict=font2)
plt.xlabel("Type of cars",fontdict=font1)
plt.ylabel("Count of cars",fontdict=font1)
plt.bar(x, y)
plt.show() 

subset = df[df['doors'] == 4]
count = subset['maint'].value_counts()["vhigh"]

subset1 = df[df['doors'] == 4]
count1 = subset1['maint'].value_counts()["high"]

subset2 = df[df['doors'] == 4]
count2 = subset2['maint'].value_counts()["med"]
subset3 = df[df['doors'] == 4]
count3 = subset3['maint'].value_counts()["low"]

x5 = np.array(["vhigh",'high','med','low'])
y5 = np.array([count, count1, count2, count3])
plt.title("Maintenance Price Comparison for 4 Doored Cars",fontdict=font1,loc = 'left')
plt.xlabel("Maintanence Price",fontdict=font2)
plt.ylabel("Cars Count",fontdict=font2)
plt.plot(x5, y5)
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()

# Check for invalid values before plotting
df['buying'] = df['buying'].astype(str)
value_counts2 = df['buying'].value_counts()
countbuyhigh = value_counts2.get('high', 0)
countbuymid = value_counts2.get('med', 0)
countbuyvhigh = value_counts2.get('vhigh', 0)

x4 = np.array([countbuyhigh, countbuymid, countbuyvhigh])
labels2 = ["High", "Medium", "Very High"]
plt.title('Buying Price Analysis', fontdict=font2)
plt.pie(x4, labels=labels2)
plt.show()

# Check for invalid values before plotting
value_counts = df['class'].value_counts()
countaccept = value_counts.get('acceptable', 0)
countunaccept = value_counts.get('unacceptable', 0)
print("Total number of acceptable cars:", countaccept)
print("Total number of unacceptable cars:", countunaccept)
y = np.array([countaccept, countunaccept])
mylabels = ["Acceptable Cars", "Unacceptable Cars"]
plt.title('Car Acceptability', fontdict=font1)
plt.pie(y, labels=mylabels)
plt.show()













