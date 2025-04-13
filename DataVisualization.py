import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setting a consistent style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

df = pd.read_csv('predictions_data.csv')

# 1. Salary Distribution for Placed Students
plt.figure()
placed_df = df[df['PlacementStatus'] == 1]
ax = sns.violinplot(x='PlacementStatus', y='salary', data=placed_df, inner='quartile')
ax.set(xlabel='', ylabel='Salary (₹)', title='Salary Distribution for Placed Students')
plt.xticks([0], ['Placed'])
plt.axhline(y=placed_df['salary'].median(), color='r', linestyle='--', label=f'Median: {placed_df["salary"].median():,.0f}')
plt.legend()
plt.tight_layout()
plt.savefig('plottings/salary_distribution.png', dpi=300)
plt.show()

# 2. CGPA Distribution with placement status comparison
plt.figure()
ax = sns.histplot(data=df, x='CGPA', hue='PlacementStatus', kde=True, 
                 bins=20, element='step', common_norm=False)
ax.set(xlabel='CGPA', ylabel='Count', title='CGPA Distribution by Placement Status')
labels = ['Not Placed', 'Placed']
plt.legend(labels=labels, title='Status')
plt.tight_layout()
plt.savefig('plottings/cgpa_distribution.png', dpi=300)
plt.show()

# 3. Correlation Heatmap
plt.figure()
correlation_cols = ['CGPA', 'Major Projects', 'Workshops/Certifications', 'Mini Projects',
                   'Skills', 'Communication Skill Rating', 'Internship', 'Hackathon',
                   '12th Percentage', '10th Percentage', 'backlogs', 'salary']
corr_matrix = df[correlation_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                mask=mask, vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
ax.set_title('Correlation Heatmap of Placement Factors', fontsize=14)
plt.tight_layout()
plt.savefig('plottings/correlation_heatmap.png', dpi=300)
plt.show()

# 4. Pairplot for key features by placement status
pair_cols = ['CGPA', 'Major Projects', 'Internship', 'Hackathon', 'PlacementStatus']
pair_df = df[pair_cols].copy()
g = sns.pairplot(pair_df, hue='PlacementStatus', diag_kind='kde', 
                plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'w'},
                diag_kws={'fill': True, 'alpha': 0.6})
g.fig.suptitle('Relationships Between Key Placement Factors', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig('plottings/key_factors_pairplot.png', dpi=300)
plt.show()

# 5. Placement Status Distribution with percentages
plt.figure()
total = len(df)
ax = sns.countplot(x='PlacementStatus', data=df)
ax.set(xlabel='Placement Status', ylabel='Count', title='Student Placement Distribution')
plt.xticks([0, 1], ['Not Placed', 'Placed'])

# Add percentage labels
for p in ax.patches:
    height = p.get_height()
    percentage = f'{height/total*100:.1f}%'
    ax.text(p.get_x() + p.get_width()/2., height + 5, percentage, ha='center')

plt.tight_layout()
plt.savefig('plottings/placement_distribution.png', dpi=300)
plt.show()

# 6. Placement by Company Type 
plt.figure()
company_order = df['company_type'].value_counts().index
ax = sns.countplot(x='company_type', data=df, hue='PlacementStatus', order=company_order)
ax.set(xlabel='Company Type', ylabel='Count', title='Placement Status by Company Type')
plt.legend(title='Status', labels=['Not Placed', 'Placed'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plottings/placement_by_company.png', dpi=300)
plt.show()

# 7. Skills Impact on Placement (Box plot)
plt.figure()
ax = sns.boxplot(x='PlacementStatus', y='Skills', data=df)
ax.set(xlabel='Placement Status', ylabel='Skills Count', title='Impact of Skills on Placement')
plt.xticks([0, 1], ['Not Placed', 'Placed'])
plt.tight_layout()
plt.savefig('plottings/skills_impact.png', dpi=300)
plt.show()

# 8. Placement Rate by CGPA Range
plt.figure()
df['CGPA_Range'] = pd.cut(df['CGPA'], bins=[5, 6, 7, 8, 9, 10], labels=['5-6', '6-7', '7-8', '8-9', '9-10'])
placement_by_cgpa = df.groupby('CGPA_Range')['PlacementStatus'].mean() * 100
ax = placement_by_cgpa.plot(kind='bar', color='skyblue')
ax.set(xlabel='CGPA Range', ylabel='Placement Rate (%)', title='Placement Success Rate by CGPA Range')
plt.xticks(rotation=0)

# Add percentage labels
for i, v in enumerate(placement_by_cgpa):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('plottings/placement_by_cgpa.png', dpi=300)
plt.show()

# 9. Salary prediction based on CGPA (scatter with regression line)
plt.figure()
ax = sns.regplot(x='CGPA', y='salary', data=placed_df, scatter_kws={'alpha':0.5})
ax.set(xlabel='CGPA', ylabel='Salary (₹)', title='Relationship Between CGPA and Salary')
plt.tight_layout()
plt.savefig('plottings/cgpa_salary_relationship.png', dpi=300)
plt.show()

# 10. Interactive plot for factors influencing placement
plt.figure(figsize=(16, 10))
factors = ['CGPA', 'Major Projects', 'Workshops/Certifications', 'Mini Projects',
          'Skills', 'Communication Skill Rating', 'Internship', 'Hackathon']

for i, factor in enumerate(factors):
    plt.subplot(2, 4, i+1)
    sns.boxplot(x='PlacementStatus', y=factor, data=df)
    plt.xlabel('Placement Status')
    plt.ylabel(factor)
    plt.xticks([0, 1], ['Not Placed', 'Placed'])
    plt.title(f'{factor} Impact')

plt.suptitle('Factors Influencing Placement', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('plottings/placement_factors.png', dpi=300)
plt.show()
