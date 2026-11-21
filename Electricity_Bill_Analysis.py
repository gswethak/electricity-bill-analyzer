# ======================================================================
# ⚡ Electricity Bill Analysis — Mini Data Analysis Project
# Runs directly in VS Code (Python extension: click "Run Cell" above any
# "# %%" marker, or click "Run" at the top to execute the whole file).
# ======================================================================

# %% [1] Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
plt.rcParams['figure.dpi'] = 100
if 'seaborn-v0_8-whitegrid' in plt.style.available:
    plt.style.use('seaborn-v0_8-whitegrid')

print("Libraries imported successfully ✅")

# %% [2] Load Dataset
# Put "Household_energy_bill_data.csv" in the SAME FOLDER as this script,
# or edit CSV_PATH below to point to wherever the file actually is.
CSV_PATH = "Household_energy_bill_data.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find '{CSV_PATH}'. Put the CSV next to this script, "
        f"or change CSV_PATH to the full path, e.g. r'C:\\Users\\you\\Downloads\\Household_energy_bill_data.csv'"
    )

df = pd.read_csv(CSV_PATH)

print("Dataset loaded successfully ✅")
print("Shape of dataset (rows, columns):", df.shape)
print(df.head(10))

# %% [3] Inspect Dataset Structure
print("Columns:", list(df.columns))
df.info()
print(df.describe())

# Column meaning:
# num_rooms, num_people, housearea, is_ac, is_tv, is_flat,
# ave_monthly_income, num_children, is_urban, amount_paid (bill in Rs.)

# %% [4] Data Cleaning
print("Missing values per column:")
print(df.isnull().sum())

print("Number of duplicate rows:", df.duplicated().sum())

cols_should_be_positive = ['num_rooms', 'num_people', 'ave_monthly_income']
for col in cols_should_be_positive:
    print(f"Invalid negative values in '{col}': {(df[col] < 0).sum()}")

df_clean = df.copy()
for col in cols_should_be_positive:
    median_val = df_clean.loc[df_clean[col] >= 0, col].median()
    df_clean.loc[df_clean[col] < 0, col] = median_val
    print(f"Replaced negative values in '{col}' with median = {median_val}")

print("Remaining negative values after cleaning:")
print((df_clean[cols_should_be_positive] < 0).sum())

print(df_clean.dtypes)
df = df_clean
print("✅ Data cleaning complete.")

# %% [5] Basic Analysis
total_bill = df['amount_paid'].sum()
avg_bill = df['amount_paid'].mean()
max_bill_row = df.loc[df['amount_paid'].idxmax()]
min_bill_row = df.loc[df['amount_paid'].idxmin()]
std_bill = df['amount_paid'].std()

print(f"💰 Total Electricity Bill: Rs. {total_bill:,.2f}")
print(f"📊 Average Bill per household: Rs. {avg_bill:,.2f}")
print(f"📈 Highest Bill: Rs. {max_bill_row['amount_paid']:,.2f} (Household #{max_bill_row.name})")
print(f"📉 Lowest Bill: Rs. {min_bill_row['amount_paid']:,.2f} (Household #{min_bill_row.name})")
print(f"📐 Std. Deviation: Rs. {std_bill:,.2f}")
print(f"🏠 Total households: {len(df)}")

# %% [6] Appliance Analysis
ac_impact = df.groupby('is_ac')['amount_paid'].mean().rename({0: 'No AC', 1: 'Has AC'})
tv_impact = df.groupby('is_tv')['amount_paid'].mean().rename({0: 'No TV', 1: 'Has TV'})

print("Average bill by AC ownership:")
print(ac_impact.round(2))
print("Average bill by TV ownership:")
print(tv_impact.round(2))

ac_extra = ac_impact['Has AC'] - ac_impact['No AC']
tv_extra = tv_impact['Has TV'] - tv_impact['No TV']
appliance_rank = pd.Series({'Air Conditioner': ac_extra, 'Television': tv_extra}).sort_values(ascending=False)

print("🔌 Appliances ranked by impact (Rs.):")
print(appliance_rank.round(2))

energy_hungry = appliance_rank.idxmax()
print(f"⚠️ Most energy-hungry appliance: {energy_hungry} (+Rs. {appliance_rank.max():,.2f})")

# %% [7] Usage Classification (Low / Moderate / High)
low_threshold = df['amount_paid'].quantile(0.33)
high_threshold = df['amount_paid'].quantile(0.66)

def classify_usage(bill):
    if bill <= low_threshold:
        return 'Low Usage'
    elif bill <= high_threshold:
        return 'Moderate Usage'
    else:
        return 'High Usage'

df['usage_class'] = df['amount_paid'].apply(classify_usage)
usage_counts = df['usage_class'].value_counts().reindex(['Low Usage', 'Moderate Usage', 'High Usage'])
print(usage_counts)

# %% [8] Warning System
mean_bill = df['amount_paid'].mean()
high_threshold_warn = mean_bill + std_bill
high_usage_households = df[df['amount_paid'] > high_threshold_warn]

print(f"Warning threshold: Rs. {high_threshold_warn:,.2f}")
print(f"Flagged households: {len(high_usage_households)}")

for idx, row in high_usage_households.head(5).iterrows():
    print(f"⚠️ WARNING: Household #{idx} has an unusually high bill of Rs. {row['amount_paid']:,.2f}.")
    print("   👉 Consider reducing the use of high-power appliances (AC, heavy electronics).")

# %% [9] Suggestions
suggestions = [
    "🔆 Use AC at 24-26°C instead of lower settings.",
    "💡 Replace old bulbs/tubelights with LED lighting.",
    "🔌 Unplug electronics (including TVs) when not in use.",
    "🌥️ Use natural ventilation/daylight where possible.",
    "🧊 Service AC units regularly.",
    "⏱️ Use timers/smart plugs on high-power appliances.",
    "📊 High Usage households should audit appliance hours.",
]
for tip in suggestions:
    print(tip)

# %% [10a] Visualization - Chart 1: Bill Bar Chart
sample = df.head(30)
plt.figure(figsize=(12, 5))
plt.bar(range(len(sample)), sample['amount_paid'], color='#3b82f6', edgecolor='black')
plt.title('Electricity Bill Amount — First 30 Households', fontsize=14, fontweight='bold')
plt.xlabel('Household Index'); plt.ylabel('Bill Amount (Rs.)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.axhline(avg_bill, color='red', linestyle='--', label=f"Average = Rs. {avg_bill:,.0f}")
plt.legend(); plt.tight_layout(); plt.show()

# %% [10b] Visualization - Chart 2: Bill Trend Line Chart
plt.figure(figsize=(12, 5))
plt.plot(df.index[:100], df['amount_paid'][:100], color='#10b981', linewidth=1.5, marker='o', markersize=3)
plt.title('Electricity Bill Trend — First 100 Households', fontsize=14, fontweight='bold')
plt.xlabel('Household Index'); plt.ylabel('Bill Amount (Rs.)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(['Bill Amount']); plt.tight_layout(); plt.show()

# %% [10c] Visualization - Chart 3: Appliance Ownership Pie Charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ac_share = df['is_ac'].value_counts().reindex([1, 0])
axes[0].pie(ac_share, labels=['Has AC', 'No AC'], autopct='%1.1f%%', colors=['#f97316', '#e5e7eb'], startangle=90, explode=(0.05, 0))
axes[0].set_title('AC Ownership Share', fontweight='bold')
tv_share = df['is_tv'].value_counts().reindex([1, 0])
axes[1].pie(tv_share, labels=['Has TV', 'No TV'], autopct='%1.1f%%', colors=['#8b5cf6', '#e5e7eb'], startangle=90, explode=(0.05, 0))
axes[1].set_title('TV Ownership Share', fontweight='bold')
plt.suptitle('Appliance Ownership Distribution', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# %% [10d] Visualization - Chart 4: Appliance Impact Ranked Bar Chart
plt.figure(figsize=(8, 5))
bars = plt.bar(appliance_rank.index, appliance_rank.values, color=['#ef4444', '#3b82f6'], edgecolor='black')
plt.title('Appliances Ranked by Impact on Average Bill', fontsize=14, fontweight='bold')
plt.xlabel('Appliance'); plt.ylabel('Extra Bill Amount (Rs.)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h, f'Rs. {h:,.0f}', ha='center', va='bottom', fontweight='bold')
plt.legend(['Extra bill vs household without the appliance']); plt.tight_layout(); plt.show()

# %% [10e] Visualization - Chart 5: Trend with Moving Average
sorted_bills = df['amount_paid'].sort_values().reset_index(drop=True)
rolling_avg = sorted_bills.rolling(window=30).mean()
plt.figure(figsize=(12, 5))
plt.plot(sorted_bills.index, sorted_bills.values, color='#94a3b8', alpha=0.5, label='Individual Household Bill (sorted)')
plt.plot(rolling_avg.index, rolling_avg.values, color='#dc2626', linewidth=2, label='30-Household Moving Average')
plt.title('Electricity Bill Trend Across Households (Sorted)', fontsize=14, fontweight='bold')
plt.xlabel('Household Rank (sorted by bill amount)'); plt.ylabel('Bill Amount (Rs.)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(); plt.tight_layout(); plt.show()

# %% [10f] Visualization - Chart 6: Low vs Moderate vs High Usage Comparison
plt.figure(figsize=(8, 5))
colors = ['#22c55e', '#eab308', '#ef4444']
bars = plt.bar(usage_counts.index, usage_counts.values, color=colors, edgecolor='black')
plt.title('Household Distribution: Low vs Moderate vs High Usage', fontsize=14, fontweight='bold')
plt.xlabel('Usage Category'); plt.ylabel('Number of Households')
plt.grid(axis='y', linestyle='--', alpha=0.6)
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h, str(int(h)), ha='center', va='bottom', fontweight='bold')
plt.legend(bars, usage_counts.index); plt.tight_layout(); plt.show()

# %% [11] Dashboard (Matplotlib Subplots)
def compute_metrics(data):
    """Recompute every KPI/statistic from the current dataframe."""
    m = {}
    m['total_bill'] = data['amount_paid'].sum()
    m['avg_bill'] = data['amount_paid'].mean()
    m['std_bill'] = data['amount_paid'].std()
    m['max_row'] = data.loc[data['amount_paid'].idxmax()]
    m['min_row'] = data.loc[data['amount_paid'].idxmin()]

    ac_imp = data.groupby('is_ac')['amount_paid'].mean().rename({0: 'No AC', 1: 'Has AC'})
    tv_imp = data.groupby('is_tv')['amount_paid'].mean().rename({0: 'No TV', 1: 'Has TV'})
    ac_ex = ac_imp.get('Has AC', 0) - ac_imp.get('No AC', 0)
    tv_ex = tv_imp.get('Has TV', 0) - tv_imp.get('No TV', 0)
    m['appliance_rank'] = pd.Series({'Air Conditioner': ac_ex, 'Television': tv_ex}).sort_values(ascending=False)

    low_t = data['amount_paid'].quantile(0.33)
    high_t = data['amount_paid'].quantile(0.66)

    def classify(x):
        if x <= low_t:
            return 'Low Usage'
        elif x <= high_t:
            return 'Moderate Usage'
        else:
            return 'High Usage'

    data['usage_class'] = data['amount_paid'].apply(classify)
    m['usage_counts'] = data['usage_class'].value_counts().reindex(
        ['Low Usage', 'Moderate Usage', 'High Usage']).fillna(0).astype(int)
    m['high_usage_households'] = data[data['amount_paid'] > m['avg_bill'] + m['std_bill']]
    return m


def draw_dashboard(data, m):
    """Redraw the full multi-panel dashboard from any dataframe + metrics dict."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('⚡ ELECTRICITY BILL ANALYSIS DASHBOARD ⚡', fontsize=20, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    kpi_data = [
        ("Total Bill Collected", f"Rs. {m['total_bill']:,.0f}", '#3b82f6'),
        ("Average Bill / Household", f"Rs. {m['avg_bill']:,.0f}", '#10b981'),
        ("Highest Bill", f"Rs. {m['max_row']['amount_paid']:,.0f}", '#ef4444'),
    ]
    for i, (label, value, color) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, i])
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=color, alpha=0.15, edgecolor=color, linewidth=2))
        ax.text(0.5, 0.62, value, ha='center', va='center', fontsize=17, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=11, color='#374151', transform=ax.transAxes)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.bar(m['usage_counts'].index, m['usage_counts'].values, color=['#22c55e', '#eab308', '#ef4444'], edgecolor='black')
    ax1.set_title('Usage Classification', fontsize=11, fontweight='bold'); ax1.set_ylabel('Households')
    ax1.tick_params(axis='x', labelsize=8); ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.bar(m['appliance_rank'].index, m['appliance_rank'].values, color=['#ef4444', '#3b82f6'], edgecolor='black')
    ax2.set_title('Appliance Impact on Bill', fontsize=11, fontweight='bold'); ax2.set_ylabel('Extra Rs.')
    ax2.tick_params(axis='x', labelsize=8); ax2.grid(axis='y', linestyle='--', alpha=0.5)

    ax3 = fig.add_subplot(gs[1, 2])
    ac_share = data['is_ac'].value_counts().reindex([1, 0]).fillna(0)
    ax3.pie(ac_share, labels=['Has AC', 'No AC'], autopct='%1.0f%%', colors=['#f97316', '#e5e7eb'], startangle=90)
    ax3.set_title('AC Ownership', fontsize=11, fontweight='bold')

    ax4 = fig.add_subplot(gs[2, 0:2])
    sb = data['amount_paid'].sort_values().reset_index(drop=True)
    ax4.plot(sb.index, sb.values, color='#94a3b8', alpha=0.6, linewidth=1)
    ax4.plot(sb.rolling(30).mean(), color='#dc2626', linewidth=2, label='Moving Avg (30)')
    ax4.set_title('Bill Trend Across Households (Sorted)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Household Rank'); ax4.set_ylabel('Bill (Rs.)')
    ax4.legend(fontsize=8); ax4.grid(True, linestyle='--', alpha=0.5)

    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    summary_text = (
        f"📊 SUMMARY\n\nHouseholds: {len(data)}\n"
        f"Lowest Bill: Rs. {m['min_row']['amount_paid']:,.0f}\n"
        f"Std. Dev: Rs. {m['std_bill']:,.0f}\n"
        f"High-Usage Alerts: {len(m['high_usage_households'])}\n"
        f"Top Appliance: {m['appliance_rank'].idxmax()}"
    )
    ax5.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=10.5,
             transform=ax5.transAxes, bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#9ca3af'))
    plt.show()


metrics = compute_metrics(df)
draw_dashboard(df, metrics)

# %% [12] Insights
print("=" * 60)
print("📌 KEY INSIGHTS")
print("=" * 60)
print(f"1. Total billing collected across {len(df)} households: Rs. {total_bill:,.2f}")
print(f"2. Average electricity bill per household: Rs. {avg_bill:,.2f}")
print(f"3. Bill amounts range from Rs. {min_bill_row['amount_paid']:,.2f} to Rs. {max_bill_row['amount_paid']:,.2f}")
print(f"4. {energy_hungry} ownership has the biggest impact on bill amount (+Rs. {appliance_rank.max():,.2f} on average).")
print(f"5. {len(high_usage_households)} households ({len(high_usage_households) / len(df) * 100:.1f}%) flagged as unusually high consumption.")
print(f"6. Usage split — Low: {usage_counts['Low Usage']}, Moderate: {usage_counts['Moderate Usage']}, High: {usage_counts['High Usage']} households.")

corr = df[['num_rooms', 'num_people', 'housearea', 'is_ac', 'is_tv',
           'ave_monthly_income', 'num_children', 'is_urban', 'amount_paid']].corr()['amount_paid'].drop('amount_paid')
top_factor = corr.abs().idxmax()
print(f"7. Factor most correlated with electricity bill: '{top_factor}' (correlation = {corr[top_factor]:.2f}).")

# %% [13] Conclusion
print(
    "This project analyzed household electricity billing data using only NumPy, Pandas, "
    "and Matplotlib — covering data cleaning, statistical analysis, appliance-impact analysis, "
    "usage classification, an automated warning system, six visualizations, and a dashboard. "
    "Air Conditioner ownership was found to be the strongest driver of higher bills."
)

# %% [14] Add New Household Data & Refresh Analysis
def add_household(data, new_record: dict):
    """Append one new household record to the dataframe and return the updated dataframe."""
    required = ['num_rooms', 'num_people', 'housearea', 'is_ac', 'is_tv', 'is_flat',
                'ave_monthly_income', 'num_children', 'is_urban', 'amount_paid']
    missing = [c for c in required if c not in new_record]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    new_row = pd.DataFrame([new_record])[required]
    updated = pd.concat([data.drop(columns=['usage_class'], errors='ignore'), new_row], ignore_index=True)
    return updated


def add_household_interactive(data):
    """Prompts you to type in values for a new household, one field at a time (terminal input)."""
    print("Enter details for the new household:")
    record = {
        'num_rooms': int(input("Number of rooms: ")),
        'num_people': int(input("Number of people: ")),
        'housearea': float(input("House area (sq.ft): ")),
        'is_ac': int(input("Has AC? (1=Yes, 0=No): ")),
        'is_tv': int(input("Has TV? (1=Yes, 0=No): ")),
        'is_flat': int(input("Is a flat/apartment? (1=Yes, 0=No): ")),
        'ave_monthly_income': float(input("Average monthly income (Rs.): ")),
        'num_children': int(input("Number of children: ")),
        'is_urban': int(input("Urban area? (1=Yes, 0=No): ")),
        'amount_paid': float(input("Electricity bill amount (Rs.): ")),
    }
    return add_household(data, record)


# ---- Edit these values to your own new household, then run this cell ----
new_household = {
    'num_rooms': 3,
    'num_people': 4,
    'housearea': 850.0,
    'is_ac': 1,            # 1 = has AC, 0 = no AC
    'is_tv': 1,            # 1 = has TV, 0 = no TV
    'is_flat': 1,          # 1 = flat/apartment, 0 = independent house
    'ave_monthly_income': 32000.0,
    'num_children': 1,
    'is_urban': 1,         # 1 = urban, 0 = rural
    'amount_paid': 720.0   # electricity bill amount (Rs.)
}

df = add_household(df, new_household)
print(f"✅ New household added. Dataset now has {len(df)} households.")

metrics = compute_metrics(df)
draw_dashboard(df, metrics)

# To add a household interactively via terminal prompts instead, run:
# df = add_household_interactive(df)
# metrics = compute_metrics(df)
# draw_dashboard(df, metrics)

# To save your updated dataset (with the new households) back to a CSV file:
# df.to_csv("Household_energy_bill_data_updated.csv", index=False)
# print("Saved updated dataset ✅")
