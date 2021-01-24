#!/usr/bin/env python
# coding: utf-8

# ## Observations and Insights 

# 

# In[96]:


# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np
from sklearn import datasets

# Study data files
mouse_metadata_path = "data/Mouse_metadata.csv"
study_results_path = "data/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)


# Combine the data into a single dataset
df_combined_mouse_data = pd.merge(study_results, mouse_metadata, how="outer", on="Mouse ID")


# Display the data table for preview
df_combined_mouse_data


# In[97]:


# Checking the number of mice.
mouse_id = df_combined_mouse_data["Mouse ID"].count()
pd.DataFrame({"Total Number of Mouse ID": [mouse_id]})


# In[98]:


df_combined_mouse_data.count()


# In[99]:


# Getting the duplicate mice by ID number that shows up for Mouse ID and Timepoint. 
duplicate_rows = df_combined_mouse_data[df_combined_mouse_data.duplicated(['Mouse ID', 'Timepoint'])]
duplicate_rows


# In[100]:


# Optional: Get all the data for the duplicate mouse ID. 
all_duplicate_rows = df_combined_mouse_data[df_combined_mouse_data.duplicated(['Mouse ID','Timepoint'])]
all_duplicate_rows


# In[101]:


df_combined_mouse_data.loc[df_combined_mouse_data["Mouse ID"] =="g989"]


# In[102]:


df_clean_mouse = df_combined_mouse_data.loc[df_combined_mouse_data["Mouse ID"] !="g989"]

df_clean_mouse


# In[103]:


# Checking the number of mice in the clean DataFrame.
total_mice = df_clean_mouse["Mouse ID"].unique()
len(total_mice)


# ## Summary Statistics

# In[ ]:





# In[104]:


# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen
mean = df_clean_mouse.groupby('Drug Regimen')['Tumor Volume (mm3)'].mean()
median = df_clean_mouse.groupby('Drug Regimen')['Tumor Volume (mm3)'].median()
variance = df_clean_mouse.groupby('Drug Regimen')['Tumor Volume (mm3)'].var()
standard_deviation = df_clean_mouse.groupby('Drug Regimen')['Tumor Volume (mm3)'].std()
sem = df_clean_mouse.groupby('Drug Regimen')['Tumor Volume (mm3)'].sem()

summary_stat_mouse_df = pd.DataFrame({"Mean": mean, "Median": median, "Variance": variance,
                                   "Standard Deviation": standard_deviation, "SEM": sem})


summary_stat_mouse_df

# Use groupby and summary statistical methods to calculate the following properties of each drug regimen: 
# mean, median, variance, standard deviation, and SEM of the tumor volume. 
# Assemble the resulting series into a single summary dataframe.


# In[105]:


# Using the aggregation method, produce the same summary statistics in a single line
summary_stat_mouse_df = df_clean_mouse.groupby("Drug Regimen").agg({"Tumor Volume (mm3)":["mean", "median", "var", "std", "sem"]})

summary_stat_mouse_df


# ## Bar and Pie Charts

# In[106]:


# Generate a bar plot showing the total number of mice for each treatment throughout the course of the study using pandas. 
#reset the index, grouping by "Drug Regimen".count())
drug_count_data = pd.DataFrame(df_clean_mouse.groupby(["Drug Regimen"]).count()).reset_index()

#Put in descending order
df_drug_data = drug_count_data.sort_values(by="Drug Regimen", ascending=False)

#Create the drug count data df down to 2 columns
df_drug_data = drug_count_data[["Drug Regimen", "Mouse ID"]]
#Set Index to "Drug Regimen"
df_drug_data = df_drug_data.set_index("Drug Regimen")


# In[79]:


df_drug_data.plot(kind="bar", figsize=(10,5))


plt.title("Total Mice Treated for each Drug Regimen")
plt.show()
plt.tight_layout()


# In[107]:


# Generate a bar plot showing the total number of measurements taken on each drug regimen using pyplot.
# Create a list of the drugs
drugs_list = summary_stat_mouse_df.index.tolist()
# drugs_list

# Create the drugs into a list
drug_count = (df_combined_mouse_data.groupby(["Drug Regimen"])["Age_months"].count()).tolist()
drug_count


# Create the x_axis to be the above list
x_axis = np.arange(len(drug_count))

x_axis = drugs_list

#Create the bar chat
plt.figure(figsize=(10,5))
plt.bar(x_axis, drug_count, color='b', alpha=0.25, align="center")

plt.title("Total Mice Treated for each Drug Regimen")
plt.xlabel("Drug Regimen")
plt.ylabel("Mice Tested");


# In[108]:


#Showing the distribution of female versus male mice using pandas

df_genders = pd.DataFrame(df_clean_mouse.groupby(["Sex"]).count()).reset_index()
df_genders.head()


# In[81]:



# Generate a pie plot showing the distribution of female versus male mice using pandas

# Create the df_genders dataframe into only two columns
df_genders = df_genders[["Sex", "Mouse ID"]]
#df_genders.head()


# Create the Pir Chart and Customize
plt.figure(figsize=(10,5))
ax1 = plt.subplot(121, aspect="equal")
colors = ["lightcoral", "lightskyblue"]
explode = (0.1,0)
df_genders.plot(kind="pie", y = "Mouse ID", ax=ax1, autopct='%1.1f%%', explode=explode,
               startangle=160, shadow=False, labels=df_genders["Sex"], colors=colors,
              legend=False, fontsize=12)

plt.title("Percent of Male & Female Mice")
plt.xlabel("")
plt.ylabel("");


# In[109]:



#Generate the same pie chart from above using PYPLOT
gender_count = (df_clean_mouse.groupby(["Sex"])["Age_months"].count()).tolist()
gender_count


# In[110]:



# Add in info from the "Sex" and "Age_months" into the pie chart
labels = ["Female", "Male"]
colors = ["lightcoral", "lightskyblue"]
explode = (0.1,0)

#Create the pie char in pylpot
plt.pie(gender_count, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%", shadow=False, startangle=160)
plt.axis("equal");
plt.title("Percent of Male & Female Mice");


# ## Quartiles, Outliers and Boxplots

# In[84]:


df_clean_mouse.head()


# In[111]:


# Calculate the final tumor volume of each mouse across four of the treatment regimens:  
# Capomulin, Ramicane, Infubinol, and Ceftamin

#Sort and arrange columns: "Drug Regimen", "Mouse ID", and "Timepoint"
df_tumor = df_clean_mouse.sort_values(["Drug Regimen", "Mouse ID", "Timepoint"], ascending=True)

# Start by getting the last (greatest) timepoint for each mouse
# Merge this group df with the original dataframe to get the tumor volume at the last timepoint
df_last = df_tumor.loc[df_tumor["Timepoint"]==45]
df_last.head().reset_index()


# In[87]:


# Grab the data for each drug and create a dataframe and box and whisker chart for them

#Capomulin
df_capo_box = df_last[df_last["Drug Regimen"].isin(["Capomulin"])]
# df_capo_box.head().reset_index()
# Create Capomulin Tumor Volume (mm3) a df object
capo_object = df_capo_box.sort_values(["Tumor Volume (mm3)"], ascending=True).reset_index()
capo_object = capo_object["Tumor Volume (mm3)"]
# capo_object
#IQR ranges for "Capomulin"
quartiles = capo_object.quantile([.25,.5,.75]).round(2)
lowerq = quartiles[0.25].round(2)
upperq = quartiles[0.75].round(2)
iqr = upperq - lowerq
lower_bound = lowerq - (1.5*iqr).round(2)
upper_bound = upperq + (1.5*iqr).round(2)

print(f'''
        The Lower Quartile of temperature in Capomulin Regimen is: {lowerq}, 
        The Upper Quartile of temperature in Capomulin Regimen is: {upperq}, 
        The IQR of temperature in Capomulin Regimen is: {iqr}, 
        The Median of temperatures for Capomulin Regimen is: {quartiles[0.5]}
        Values below {lower_bound} could be the outliers in Capomulin Regimen
        Values below {upper_bound} could be the outliers in Capomulin Regimen
        ''')


# In[112]:


# Grab the data for each drug and create a dataframe and box and whisker chart for them

#Ramicane
df_rami_box = df_last[df_last["Drug Regimen"].isin(["Ramicane"])]
# df_capo_box.head().reset_index()
# Create Capomulin Tumor Volume (mm3) a df object
rami_object = df_rami_box.sort_values(["Tumor Volume (mm3)"], ascending=True).reset_index()
rami_object = rami_object["Tumor Volume (mm3)"]
# rami_object
#IQR ranges for "Ramicane"
quartiles = rami_object.quantile([.25,.5,.75]).round(2)
lowerq = quartiles[0.25].round(2)
upperq = quartiles[0.75].round(2)
iqr = upperq - lowerq
lower_bound = lowerq - (1.5*iqr).round(2)
upper_bound = upperq + (1.5*iqr).round(2)

print(f'''
        The Lower Quartile of temperature in the Ramicane Regimen is: {lowerq}, 
        The Upper Quartile of temperature in the Ramicane Regimen is: {upperq}, 
        The IQR of temperature in the Ramicane Regimen is: {iqr}, 
        The Median of temperatures for the Ramicane Regimen is: {quartiles[0.5]}
        Values below {lower_bound} could be the outliers in the Ramicane Regimen
        Values below {upper_bound} could be the outliers in the Ramicane Regimen
        ''')


# In[89]:


# Grab the data for each drug and create a dataframe and box and whisker chart for them

#Infubinol
df_infu_box = df_last[df_last["Drug Regimen"].isin(["Infubinol"])]
# df_capo_box.head().reset_index()
# Create Capomulin Tumor Volume (mm3) a df object
infu_object = df_infu_box.sort_values(["Tumor Volume (mm3)"], ascending=True).reset_index()
infu_object = infu_object["Tumor Volume (mm3)"]
# infu_object
#IQR ranges for "Infubinol"
quartiles = infu_object.quantile([.25,.5,.75]).round(2)
lowerq = quartiles[0.25].round(2)
upperq = quartiles[0.75].round(2)
iqr = upperq - lowerq
lower_bound = lowerq - (1.5*iqr).round(2)
upper_bound = upperq + (1.5*iqr).round(2)

print(f'''
        The Lower Quartile of temperature in the Infubinol Regimen is: {lowerq}, 
        The Upper Quartile of temperature in the Infubinol Regimen is: {upperq}, 
        The IQR of temperature in the Infubinol Regimen is: {iqr}, 
        The Median of temperatures for the Infubinol Regimen is: {quartiles[0.5]}
        Values below {lower_bound} could be the outliers in the Infubinol Regimen
        Values below {upper_bound} could be the outliers in the Infubinol Regimen
        ''')


# In[113]:


# Grab the data for each drug and create a dataframe and box and whisker chart for them

#Ceftamin
df_ceft_box = df_last[df_last["Drug Regimen"].isin(["Ceftamin"])]
# df_capo_box.head().reset_index()
# Create Capomulin Tumor Volume (mm3) a df object
ceft_object = df_ceft_box.sort_values(["Tumor Volume (mm3)"], ascending=True).reset_index()
ceft_object = ceft_object["Tumor Volume (mm3)"]
# ceft_object
#IQR ranges for "Ceftamin"
quartiles = ceft_object.quantile([.25,.5,.75]).round(2)
lowerq = quartiles[0.25].round(2)
upperq = quartiles[0.75].round(2)
iqr = upperq - lowerq
lower_bound = lowerq - (1.5*iqr).round(2)
upper_bound = upperq + (1.5*iqr).round(2)

print(f'''
        The Lower Quartile of temperature in the Ceftamin Regimen is: {lowerq}, 
        The Upper Quartile of temperature in the Ceftamin Regimen is: {upperq}, 
        The IQR of temperature in the Ceftamin Regimen is: {iqr}, 
        The Median of temperatures for the Ceftamin Regimen is: {quartiles[0.5]}
        Values below {lower_bound} could be the outliers in the Ceftamin Regimen
        Values below {upper_bound} could be the outliers in the Ceftamin Regimen
        ''')


# In[114]:


# Generate a box plot of the final tumor volume of each mouse across four regimens of interest
combined_box_plot = [capo_object, rami_object, infu_object, ceft_object]

# Use the above list to plot all four boxes into fig5
fig5, ax5 = plt.subplots()
ax5.set_title("Final Tumor Volume in Multiple Drug Regimens")
ax5.set_ylabel("Final Tumor Volume (mm3)")
ax5.boxplot(combined_box_plot)
plt.show()


# ## Line and Scatter Plots

# In[127]:


# Generate a line plot of tumor volume vs. time point for a mouse treated with Capomulin
# Create a DF with just capomulin data
df_capomulin = df_clean_mouse.loc[df_combined_mouse_data["Drug Regimen"] == "Capomulin"]
df_capomulin = df_capomulin.reset_index()

#Get data for one mouse
mouse_capo = df_capomulin.loc[df_capomulin["Mouse ID"] == "m601"]
mouse_capo


# In[128]:



# Grab the timepoint and tumor volume for this one mouse, and arrange into two columns
mouse_capo = mouse_capo.loc[:, ["Timepoint", "Tumor Volume (mm3)"]]
#Reset the index 
mouse_capo = mouse_capo.reset_index(drop=True)


# Create the line plot showing the Capomulin treatment with this one mouse
mouse_capo.set_index("Timepoint").plot(figsize=(5,5), linewidth=3, color="green", title="Capomulin treatment: Mouse m601");


# In[129]:


# Generate a scatter plot of mouse weight versus average tumor volume for the Capomulin regimen
# Use the capomulin dataframe, and arrange into 3 columns "Mouse ID", "Weight (g)", and "Tumor Volume (mm3)" 
df_mouse_weight = df_capomulin.loc[:, ["Mouse ID", "Weight (g)", "Tumor Volume (mm3)"]]


# In[130]:



#Get average tumor volume for each mouse under Capomulin regimen..put into DataFrame
average_tumor_volume = pd.DataFrame(df_mouse_weight.groupby(["Mouse ID", "Weight (g)"])["Tumor Volume (mm3)"].mean()).reset_index()

#rename tumor volume (mm3) to Average Tumor Volume...rename(columns={"":""})...refer stack overflow
average_tumor_volume = average_tumor_volume.rename(columns={"Tumor Volume (mm3)": "Average Tumor Volume"})
average_tumor_volume.head()


# In[131]:


#Create the Scatter plot of the above data
average_tumor_volume.plot(kind="scatter", x="Weight (g)", y="Average Tumor Volume", figsize=(6,4), color="green", title="Mouse Weight vs. Average Tumor Volume")
plt.show()


# ## Correlation and Regression

# In[120]:


# Calculate the correlation coefficient and linear regression model 
# for mouse weight and average tumor volume for the Capomulin regimen
# Calculate the correlation coefficient and linear regression model 
# for mouse weight and average tumor volume for the Capomulin regimen
mice_weight = average_tumor_volume.iloc[:,2]
tumor_volume = average_tumor_volume.iloc[:,1]
#compute Pearson correlation between the two
correlation = st.pearsonr(mice_weight,tumor_volume)
print(f'''
The Correlation Coefficient between mouse weight and tumor volume is {round(correlation[0],2)}
''')


# In[134]:


from scipy.stats import linregress


# In[135]:


# Create a line of regression for 

# Add the linear regression equation and line to plot

x_values = average_tumor_volume['Weight (g)']
y_values = average_tumor_volume['Average Tumor Volume']
(slope, intercept, rvalue, pvalue, stderr) = linregress(x_values, y_values)
regress_values = x_values * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x +" + str(round(intercept,2))
plt.scatter(x_values, y_values)
plt.plot(x_values, regress_values,"r-")
plt.annotate(line_eq,(6,5), fontsize=12)
plt.xlabel("Mouse Weight")
plt.ylabel("Average Tumor Volume")
plt.title("Weight vs Average Tumor Volume")
plt.show()


# In[ ]:




