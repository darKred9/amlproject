import pandas as pd
import altair as alt # for simplicity

from utils.path_utils import get_path_from_project_root

def analyze_label_distribution(file_path):
    df = pd.read_csv(file_path)
    
    label_distribution = df['y'].value_counts(normalize=True) * 100
    label_df = pd.DataFrame({
        'Label': ['No (0)', 'Yes (1)'],
        'Percentage': [label_distribution.get(0, 0), label_distribution.get(1, 0)]
    })
    
    chart = alt.Chart(label_df).mark_bar().encode(
        x=alt.X('Label:N', title='Label Value'),
        y=alt.Y('Percentage:Q', title='Percentage (%)'),
        color=alt.Color('Label:N', scale=alt.Scale(range=['#1f77b4', '#ff7f0e'])), # blue + orange
        tooltip=['Label', 'Percentage']
    ).properties(
        title='Dataset Label (y) Distribution',
        width=400,
        height=300
    )
    
    text = chart.mark_text(
        align='center',
        baseline='middle',
        dy=-10,
        fontSize=14
    ).encode(
        text=alt.Text('Percentage:Q', format='.2f')
    )
    
    final_chart = (chart + text).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16
    )
    
    return final_chart

if __name__ == "__main__":
    
    # Group 1
    file_path = get_path_from_project_root("data", "processed", "bank_g1.csv")
    chart = analyze_label_distribution(file_path)
    
    save_path = get_path_from_project_root("results", "data_overview", "distribution_g1.html")
    chart.save(save_path)


    # Group 2
    file_path = get_path_from_project_root("data", "processed", "bank_g2.csv")
    chart = analyze_label_distribution(file_path)
    
    save_path = get_path_from_project_root("results", "data_overview", "distribution_g2.html")
    chart.save(save_path)

    # Group 3
    file_path = get_path_from_project_root("data", "processed", "bank_g3.csv")
    chart = analyze_label_distribution(file_path)
    
    save_path = get_path_from_project_root("results", "data_overview", "distribution_g3.html")
    chart.save(save_path)

    # Group 4
    file_path = get_path_from_project_root("data", "processed", "bank_g4.csv")
    chart = analyze_label_distribution(file_path)
    
    save_path = get_path_from_project_root("results", "data_overview", "distribution_g4.html")
    chart.save(save_path)