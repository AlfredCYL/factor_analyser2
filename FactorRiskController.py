import pandas as pd
import numpy as np
import math
from joypy import joyplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import cm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100 # 设置分辨率

class FactorRiskController():
    def __init__(self, factor, bt_data, time_point = '1100', period = 1, stk_pool = None, del_st = True, del_lowPri=True, del_newList = True):
        # Filter data according to required conditions
        conditions = []
        if del_st:
            conditions.append(~bt_data['is_st'].values)
        if del_lowPri:
            conditions.append(bt_data['pre_vwap'].values > 3)
        if del_newList:
            conditions.append(bt_data['listed_days'].values > 30)
        if stk_pool:
            if stk_pool in bt_data.columns:
                conditions.append(bt_data[stk_pool].fillna(0).values.astype(bool))
            else:
                raise ValueError(f"Index {stk_pool} not Found in bt_data.")
        if conditions:
            mask = np.logical_and.reduce(conditions)
            bt_data = bt_data.loc[mask]
            
        self._bt_data = bt_data # 存储回测数据
        self._stack_to_BTdata(factor,'factor_data') # 合并因子数据
        
        self._default_time_point = time_point
        self._default_period = period
        self._default_returns_column = f"fwd_ret{period}d_shift0_{time_point}"

    ######## Helper Function ##########
    def _stack_to_BTdata(self, df, col_name, how = 'inner'): # stack n_dates * p_stocks dataframe to bt_data
        stack_df = df.stack().to_frame(col_name).reset_index(names = ['date','symbol'])
        self._bt_data = self._bt_data.merge(stack_df,on = ['date','symbol'], how = how)
    
    def _unstack_from_BTdata(self, col_name):
        unstack_df = self._bt_data.pivot(index='date', columns='symbol', values=col_name)
        return unstack_df
    
    def _cross_standardize_BTdata(self, col_name):
        # 注意 在选定的股票池中 cross standardize 并不等价于股票在全市场中crs的结果
        def cross_standardize(df):
            mean = df.mean(axis=1)
            std = df.std(axis=1)
            standardized_df = df.sub(mean, axis=0).div(std, axis=0)
            return standardized_df
        
        CRS_name = f"{col_name}_CRS" # CRS stands for cross standardize
        if CRS_name in self._bt_data.columns: # 若已有数据 无需重复计算
            return 
            
        df = self._unstack_from_BTdata(col_name)
        CRS_res = cross_standardize(df)
        self._stack_to_BTdata(CRS_res,CRS_name)
    
    def _cross_qcut_BTdata(self, col_name, n_groups):
         # 注意  ['Size%5', 'Liquidity%5', 'pre_vwap%5', 'pre_amount%5', 'pre_ret1d_shift0_1500%5', 'pre_ret1d_std5d%5'] 目前六个默认的指标是在全市场的股票池子中进行的分组
        # 除此之外的其他qcut 是在选定的股票池中重新分组 与全市场分组并不完全等价
        def numpy_qcut(row, n_groups):
            valid_data = row[~np.isnan(row)]  # 去除 NaN 值
            quantiles = np.percentile(valid_data, np.linspace(0, 100, n_groups + 1))[0:-1]  # 计算分位数
            group = np.full(row.shape, np.nan)  # 初始化为 NaN 的浮点数组
            group[~np.isnan(row)] = np.searchsorted(quantiles, row[~np.isnan(row)], side="right") - 1 # 分配到各个分位
            return group
            
        qcut_name = f"{col_name}%{n_groups}"
        if qcut_name in self._bt_data.columns: # 若已有数据 无需重复计算
            return 
            
        df = self._unstack_from_BTdata(col_name)
        col_groups = np.apply_along_axis(numpy_qcut, 1, df.values, n_groups) + 1 # 从 1 开始分组
        qcut_res = pd.DataFrame(col_groups, index=df.index, columns=df.columns) 
        self._stack_to_BTdata(qcut_res,qcut_name)

    ######## 分组属性分析 ##########
    def plot_grouped_style_ts(self, style_list, n_groups = 5, n_cols = 3):
        """
        Plot grouped style analysis with each style factor grouped by factor values.
    
        Parameters:
        - style_list: List of style factor columns to plot
        - n_groups: Number of groups to split the factor into (default: 10)
        - n_cols: Number of columns for the subplot layout (default: 3)
        """
        group_key = f"factor_data%{n_groups}"

        # Ensure group key is in in bt_data
        self._cross_qcut_BTdata('factor_data',n_groups)
            
        # Calculate the number of rows needed for the given number of columns
        n_styles = len(style_list)
        n_rows = math.ceil(n_styles / n_cols)
    
        # Prepare the plot
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, (15/n_cols) / 3 * n_rows), sharex=True)
        fig.subplots_adjust(hspace=0.4, wspace=0.2)
    
        # Flatten axes for easy indexing and handle cases with less than n_cols styles
        axes = axes.flatten()
    
        # Define color map with RdYlBu, emphasizing the first and last groups
        colors = cm.RdYlBu(np.linspace(0, 1, n_groups))
    
        grouped = self._bt_data.groupby(['date', group_key])
        # Process each style factor
        for i, style in enumerate(style_list):
            # Group by date and group to calculate mean values efficiently
            grouped_mean = grouped[style].mean().unstack()
    
            # Plot each group as a line in the corresponding subplot
            for group in grouped_mean.columns:
                group = int(group)
                linewidth = 1.5 if (group-1) in [0, n_groups - 1] else 0.8  # Thicker for first and last groups
                alpha = 0.9 if (group-1) in [0, n_groups - 1] else 0.8
                axes[i].plot(grouped_mean.index, grouped_mean[group], label=str(group),
                             color=colors[group - 1], linewidth=linewidth, alpha=alpha)
            
            # Set title for each subplot
            axes[i].set_title(style)
        
        # Set common x and y labels
        fig.text(0.5, 0.04, 'Date', ha='center', va='center', fontsize=12)
        fig.text(0.04, 0.5, 'Style Value', ha='center', va='center', rotation='vertical', fontsize=12)
    
        # Add a single legend outside the plot
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Groups", loc='upper right', bbox_to_anchor=(1.01, 0.9), frameon=False)
    
        # Hide any unused subplots if n_styles is not a multiple of n_cols
        for j in range(n_styles, n_rows * n_cols):
            fig.delaxes(axes[j])
        
        plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.96])
        plt.suptitle("Style Exposure for Each Factor Group")
        plt.show()

    def plot_grouped_style(self, style_list, n_groups = 10, CRS = True):
        """
        Plot a line chart of pre-standardized style values across factor-defined groups.
    
        Parameters:
        - style_list: List of style factor columns to plot
        - n_groups: Number of groups to split the factor into (default: 10)
        - CRS: Whether to cross standardize style_values (default: True)
        """
        group_key = f"factor_data%{n_groups}"

        # Ensure group key is in in bt_data
        self._cross_qcut_BTdata('factor_data',n_groups)

        if CRS:
            for i in style_list:
                self._cross_standardize_BTdata(i)
            style_list = [f"{i}_CRS" for i in style_list] 
            
        # Prepare data for plotting by calculating group means
        group_means = self._bt_data.groupby(group_key, observed=False)[style_list].mean()
        
        # Plot
        plt.figure(figsize=(15, 5))
        for style in style_list:
            plt.plot(group_means.index, group_means[style], marker='o', label=style)
        
        # Set titles and labels
        plt.title('Style Values Across Factor Groups')
        plt.xlabel('Group')
        plt.ylabel('Style Value')
        plt.legend(title="Styles", bbox_to_anchor=(1,1), frameon=False, loc='upper left')
        plt.grid(True,axis = 'both',which = 'both')
        plt.xticks(group_means.index)
        plt.show()

    ######## 分类IC分析 ##########
    def _parse_category_col(self, input_data):
        cat_name = None
        if isinstance(input_data, str):
            cat_name = input_data
        elif isinstance(input_data, tuple) and len(input_data) == 2 and isinstance(input_data[0], str) and isinstance(input_data[1], int):
            # Ensure group key is in in bt_data
            self._cross_qcut_BTdata(input_data[0],input_data[1])
            cat_name = f"{input_data[0]}%{input_data[1]}"
        else:
            raise ValueError("List elements must be either strings or tuples of (str, int)")
        return cat_name
                
    def plot_category_cumIC(self, category_column):
        """
        Plot cumulative Rank IC and horizontal Rank ICIR distribution for each category with filtering and dynamic text alignment.
    
        Parameters:
        - category_column: A list containing strings or tuples. The tuple format is (str, int).
        """
        cat_name = self._parse_category_col(category_column)
        
        # Initialize dictionaries to store IC values and cumulative ICs
        rank_ic = {}
        cum_rank_ic = {}
    
        # Calculate Rank IC for each category
        for category, group_data in self._bt_data.groupby(cat_name):
            # Calculate Spearman Rank IC for each date in the category
            ic = group_data.groupby('date', group_keys=False).apply(
                lambda x: x['factor_data'].corr(x[self._default_returns_column ], method='spearman')
            )
            rank_ic[category] = ic.dropna()  # Drop any NaN values from missing correlations
            cum_rank_ic[category] = rank_ic[category].cumsum()
    
        # Prepare to plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Determine categories to plot based on mean IC if there are more than 6 categories
        if len(rank_ic) > 6:
            sorted_by_mean_ic = sorted(rank_ic.items(), key=lambda x: x[1].mean(), reverse=False)
            top_categories = sorted_by_mean_ic[:3]  # Top 3 by mean IC
            bottom_categories = sorted_by_mean_ic[-3:]  # Bottom 3 by mean IC
            categories_to_plot = top_categories + bottom_categories
        else:
            categories_to_plot = rank_ic.items()  # All categories if 6 or fewer
    
        # Plot cumulative Rank IC on the left
        for category, ic_values in categories_to_plot:
            axes[0].plot(cum_rank_ic[category].index, cum_rank_ic[category].values,
                         label=f"{category} - Mean={ic_values.mean():.2f}, "
                               f"Std={ic_values.std():.2f}, "
                               f"ICIR={ic_values.mean() / ic_values.std():.2f}")
        axes[0].set_title(f'Cumulative Rank IC by {cat_name}')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Cumulative Rank IC')
        axes[0].legend(loc='best', fontsize='small')
        axes[0].grid(True)
    
        # Calculate Rank ICIR for each category and rank by ICIR in descending order
        rank_icir = {cat: ic.mean() / ic.std() for cat, ic in rank_ic.items() if ic.std() > 0}
    
        # Extract sorted categories and ICIR values
        categories = [item[0] for item in rank_icir.items()]
        icir_values = [item[1] for item in rank_icir.items()]
    
        # Plot horizontal Rank ICIR distribution
        bars = axes[1].barh(categories, icir_values, color='skyblue', edgecolor = '#A9A9A9')
        
        # Add individual statistical summary as text labels to the right of each bar
        for i, bar in enumerate(bars):
            category = categories[i]
            icir_value = icir_values[i]
            text_position = bar.get_width() + 0.005 if icir_value > 0 else bar.get_width() - 0.005
            axes[1].text(text_position, bar.get_y() + bar.get_height() / 2, f"{icir_value:.2f}",
                         ha='left' if icir_value > 0 else 'right', va='center', fontsize=8)
    
        # Formatting for horizontal Rank ICIR plot
        axes[1].set_title(f'Rank ICIR Distribution by {cat_name}')
        axes[1].set_xlabel('Rank ICIR')
        axes[1].set_ylabel('Category')
        axes[1].set_yticks(categories)  # 确保每个类别都有对应的 y 轴标签
        # axes[1].grid(True)
    
        plt.tight_layout()
        plt.show()

    def plot_category_IC(self, category_column):
        """
        Plot IC distribution as a joyplot for each category.
    
        Parameters:
        - category_column: A list containing strings or tuples. The tuple format is (str, int).
        """
        cat_name = self._parse_category_col(category_column)
        
        # Initialize a dictionary to store IC values for each category
        rank_ic = {}
    
        # Calculate Rank IC for each category
        for category, group_data in self._bt_data.groupby(cat_name):
            # Calculate Spearman Rank IC for each date in the category
            ic = group_data.groupby('date', group_keys=False).apply(
                lambda x: x['factor_data'].corr(x[self._default_returns_column], method='spearman')
            )
            rank_ic[category] = ic.dropna()  # Drop any NaN values from missing correlations
    
        # Create a DataFrame for joyplot
        ic_data = pd.DataFrame({
            'IC': np.concatenate(list(rank_ic.values())),
            'Category': np.repeat(list(rank_ic.keys()), [len(v) for v in rank_ic.values()])
        })
    
        # Plot IC distribution as a joyplot
        joyplot(ic_data, by="Category", column="IC", grid=True, linewidth=1, fade=True, figsize=(15, 5))
        plt.title(f"IC Distribution by {cat_name}")
        plt.xlabel("IC Value")
        plt.tight_layout()
        plt.show()

    ######## 多空分析 ##########
    def _calculate_top_bottom_info(self, n, columns_list):
        factor_returns = []
        top_bottom_info = {}
    
        # Group by date to calculate daily factor returns
        for date, group_data in self._bt_data.groupby('date'):
            # Sort by factor value
            sorted_data = group_data.sort_values(by='factor_data', ascending=False)
            
            # Select top n and bottom n stocks
            top_n = sorted_data.head(n)
            bottom_n = sorted_data.tail(n)
            
            # Calculate long and short returns
            top_return = top_n[self._default_returns_column].mean()  
            bottom_return = bottom_n[self._default_returns_column].mean()
            
            # Calculate factor return as the difference
            factor_return = top_return - bottom_return
            factor_returns.append({'date': date, 'top_bottom_return': factor_return})
            
            # Store top and bottom n stock info for this date
            top_bottom_info[date] = {
                'Top N': top_n[columns_list],
                'Bottom N': bottom_n[columns_list]
            }
    
        # Convert factor returns to DataFrame
        factor_returns = pd.DataFrame(factor_returns)
        factor_returns['top_bottom_return'] = (factor_returns['top_bottom_return'].astype(float) + 1).cumprod() 
        return factor_returns, top_bottom_info

    def _summarize_top_bottom_info(self, top_bottom_info, columns_list):
        # Concatenate all top and bottom data into a single DataFrame for each
        top_all = pd.concat([info['Top N'] for info in top_bottom_info.values()], ignore_index=True)
        bottom_all = pd.concat([info['Bottom N'] for info in top_bottom_info.values()], ignore_index=True)
        
        # Define aggregation rules
        agg_rules = {}
        for col in columns_list:
            if col == 'symbol':
                continue
            elif col == 'date':
                agg_rules[col] = 'count'  # Count appearances
            elif pd.api.types.is_numeric_dtype(top_all[col]) or pd.api.types.is_bool_dtype(top_all[col]) :
                agg_rules[col] = 'mean'  # Take mean for numerical or boolean columns
            else:
                agg_rules[col] = 'last'  # Take last for other columns
        
        # Calculate appearance counts and mean values for top stocks
        top_summary = (
            top_all.groupby('symbol')
            .agg(agg_rules)
            .rename(columns={'date': 'count'})  # Rename date count to 'count'
            .sort_values('count', ascending=False)
        )
    
        # Calculate appearance counts and mean values for bottom stocks
        bottom_summary = (
            bottom_all.groupby('symbol')
            .agg(agg_rules)
            .rename(columns={'date': 'count'})  # Rename date count to 'count'
            .sort_values('count', ascending=False)
        )
    
        return top_summary, bottom_summary
    
    def top_bottom_analysis(self, n=10, columns_list=[], plot = True, save_html = True, output_file="./top_bottom_analysis.html"):
        """
        Plot factor returns with details for top and bottom n stocks at each date.
    
        Parameters:
        - n: Number of stocks for checking.
        - columns_list: Column names to count additional to ['date','symbol','secu_abbr','factor_data'].
        - columns_list: List of additional column names to display in hover text.
        - plot: Whether to plot top_bottom return with detials. (Default: True)
        - output_file: HTML file to save the plot.
        - save_html: Whether to save html file. (Default: True)
        """
        base_list = ['date','symbol','secu_abbr','factor_data'] 
        columns_list = base_list + [i for i in columns_list if i not in base_list]
        
        factor_returns, top_bottom_info = self._calculate_top_bottom_info(n = n, columns_list = columns_list)
        
        if plot or save_html:
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            # Add factor return line
            fig.add_trace(go.Scatter(x=factor_returns['date'], y=factor_returns['top_bottom_return'],
                                     mode='lines', name='Top-Bottom Return', line=dict(width=3) ))
        
            # Define hover data for top and bottom stocks
            hover_texts = []
            for date in factor_returns['date']:
                if date in top_bottom_info:
                    top_df = top_bottom_info[date]['Top N']
                    bottom_df = top_bottom_info[date]['Bottom N']
                    
                    # Generate table header for hover text
                    header_text = " | ".join(columns_list)
                    
                    # Prepare hover text for top and bottom stocks with type-specific formatting
                    top_text = f"<b>Top {n} Stocks</b><br><u>{header_text}</u><br>" + "<br>".join(
                        [" | ".join(
                            f"{row[col].strftime('%Y-%m-%d')}" if isinstance(row[col], pd.Timestamp) else
                            f"{row[col]:.3f}" if isinstance(row[col], float) else
                            f"{row[col]}" 
                            for col in columns_list
                        ) for _, row in top_df.iterrows()]
                    )
                    
                    bottom_text = f"<b>Bottom {n} Stocks</b><br><u>{header_text}</u><br>" + "<br>".join(
                        [" | ".join(
                            f"{row[col].strftime('%Y-%m-%d')}" if isinstance(row[col], pd.Timestamp) else
                            f"{row[col]:.3f}" if isinstance(row[col], float) else
                            f"{row[col]}"
                            for col in columns_list
                        ) for _, row in bottom_df.iterrows()]
                    )
        
                    hover_texts.append(f"<b>{date}</b><br><br>{top_text}<br><br>{bottom_text}")
                else:
                    hover_texts.append("No data available") 

            # Add hover text to factor return trace
            fig.data[0].hovertext = hover_texts
            fig.data[0].hoverinfo = "text"

            # Layout settings
            fig.update_layout(
                title="Factor Return and Stock Information",
                xaxis_title="Date",
                yaxis_title="Factor Return",
                hovermode="x"
            )

            if plot:
                fig.show()
                
            if save_html:
                fig.write_html(output_file)

        return self._summarize_top_bottom_info(top_bottom_info, columns_list)

    ######## 换手率及信号衰减分析 ##########
    def _calculate_leadlag_rankicir(self, lags):
        rankic_data = []

        # Pivot the data to have dates as rows and stocks as columns
        factor = self._unstack_from_BTdata('factor_data')

        # Calculate Rank IC for each shift by date
        for shift in lags:
            return_column = f'fwd_ret1d_shift{-shift}_{self._default_time_point}' # 注意 lag的含义是 T日的因子值 对应 T+lag日的收益 对应ret的shift要乘以-1

            # Check if the shifted return column exists
            if return_column not in self._bt_data.columns:
                raise ValueError(f"Column {return_column} not found in DataFrame.")

            # Pivot the return data to have dates as rows and stocks as columns
            return_df = self._unstack_from_BTdata(return_column)
            
            # Calculate Rank IC using corrwith
            daily_rank_ic = factor.corrwith(return_df, axis=1, method='spearman')
            
            # Store shift and daily Rank IC
            rankic_data.append({'shift': shift, 'daily_rank_ic': daily_rank_ic.dropna().values})

        # Convert to DataFrame
        rankic_df = pd.DataFrame(rankic_data)

        # Calculate ICIR (mean / std) for each shift
        rankic_df['rank_ic_mean'] = rankic_df['daily_rank_ic'].apply(np.mean)
        rankic_df['rank_ic_std'] = rankic_df['daily_rank_ic'].apply(np.std)
        rankic_df['rank_icir'] = rankic_df['rank_ic_mean'] / rankic_df['rank_ic_std']

        # Select and return only relevant columns
        return rankic_df
        
    def plot_leadlag_rankicir(self, lags=range(-10, 11)):
        """
        Plot the lead-lag Rank ICIR as a bar chart.
    
        Parameters:
        - lags: Range or list of shifts for calculating lead-lag Rank IC (e.g., range(-10, 11)).
        - return_tp: Backtest time point. (Default: 1500)
        """
        rankicir_df = self._calculate_leadlag_rankicir(lags=lags)
    
        fig, ax = plt.subplots(figsize=(15, 5))
        bars = ax.bar(rankicir_df['shift'], rankicir_df['rank_icir'], color='skyblue', edgecolor='#A9A9A9')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Rank ICIR')
        ax.set_title('Lead-Lag Rank ICIR Analysis')
        ax.grid(True, axis='both', linestyle='--', linewidth=0.5)
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
    
        # Display rank_icir values on top of the bars
        for bar in bars:
            height = bar.get_height()
            shift = bar.get_x() + bar.get_width() / 2
            if height >= 0:
                ax.text(shift, height, f'{height:.3f}', ha='center', va='bottom')
            else:
                ax.text(shift, height, f'{height:.3f}', ha='center', va='top')
    
        plt.tight_layout()
        plt.show()
        
    def _calculate_turnover_rate(self, n_groups=10):
        group_key = f"factor_data%{n_groups}"
            
        # Ensure group key is in in bt_data
        self._cross_qcut_BTdata('factor_data', n_groups)
        
        # Create a presence matrix with date as index and symbol as columns for each group
        turnover_data = {}
    
        # Iterate over each unique group to create a separate presence matrix
        for group in self._bt_data[group_key].unique():
            # Create a mask where each cell is 1 if the symbol is in the group on that date, else 0
            presence_matrix = self._bt_data[self._bt_data[group_key] == group].pivot(index='date', columns='symbol', values=group_key).notna().astype(int)
            # Store the presence matrix in a dictionary
            turnover_data[group] = presence_matrix.diff().abs().sum(axis=1) / presence_matrix.shift().sum(axis=1)
    
        # Combine turnover data into a single DataFrame
        turnover_df = pd.DataFrame(turnover_data).sort_index(axis = 1)
        return turnover_df
            
    def plot_group_turnover(self, n_groups=10, annualization = 252):
        """
        Plot turnover rate for each factor group over time in a single plot.
    
        Parameters:
        - n_groups: Number of groups to split the factor into (default: 10)
        - annualization: Annualization factor for calculating annulized turnover 
        """
        turnover_df = self._calculate_turnover_rate(n_groups)
    
        # Calculate the annulized mean turnover rate for each group
        mean_turnover = turnover_df.mean() * annualization
    
        # Plot the turnover rates
        turnover_df.plot(figsize=(15, 5), cmap='RdYlBu')
    
        # Set titles and labels
        plt.title("Daily Turnover Rate by Factor Group")
        plt.xlabel("Date")
        plt.ylabel("Turnover Rate")
    
        # Add legend outside the plot
        legend_labels = [f"{i} Mean: {mean_turnover[i]:.2f}" for i in turnover_df.columns]
        plt.legend(legend_labels, title="Groups", bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.96, 1])
        plt.show()

    def _calculate_signal_decay_probability(self, k, max_lag):
        # Pivot to create a matrix with dates as rows and symbols as columns
        factor_matrix = self._unstack_from_BTdata('factor_data').values
    
        # Calculate the thresholds for top and bottom k% for each row (date)
        top_thresholds = np.nanquantile(factor_matrix, 1 - k, axis=1, keepdims=True)
        bottom_thresholds = np.nanquantile(factor_matrix, k, axis=1, keepdims=True)
    
        # Generate presence matrices for top and bottom k% stocks
        top_presence = (factor_matrix >= top_thresholds).astype(int)
        bottom_presence = (factor_matrix <= bottom_thresholds).astype(int)
    
        # Initialize decay data storage
        decay_data = {'lag': [], 'top_k%': [], 'bottom_k%': []}
    
        # Calculate retention probability for each lag
        for lag in range(1, max_lag + 1):
            # Calculate retained top and bottom stocks for each lag
            top_retained = (top_presence[lag:] * top_presence[:-lag]).sum(axis=1) /  top_presence[:-lag].sum(1)
            bottom_retained = (bottom_presence[lag:] * bottom_presence[:-lag]).sum(axis=1) / bottom_presence[:-lag].sum(1)
    
            # Store mean retention rate for current lag
            decay_data['lag'].append(lag)
            decay_data['top_k%'].append(np.nanmean(top_retained))
            decay_data['bottom_k%'].append(np.nanmean(bottom_retained))
    
        decay_df = pd.DataFrame(decay_data)
        return decay_df
    
    def plot_signal_decay(self, k=0.1, max_lag=10):
        """
        Plot the signal decay for top and bottom k% stocks over different lags.
    
        Parameters:
        - k: Percentage of stocks to consider as top and bottom (e.g., 0.1 for 10%).
        - max_lag: Maximum lag period to calculate probabilities for.
        """
        decay_df = self._calculate_signal_decay_probability(k = k, max_lag = max_lag)
        
        x = decay_df['lag']
        width = 0.35  # width of bars
    
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Plotting the bars for top and bottom k%
        top_bars = ax.bar(x - width/2, decay_df['top_k%'], width=width, color='skyblue', label=f'Top {k*100:.1f}% Retention')
        bottom_bars = ax.bar(x + width/2, decay_df['bottom_k%'], width=width, color='salmon', label=f'Bottom {k*100:.1f}% Retention')

        # Labels and title
        ax.set_xlabel('Lag')
        ax.set_ylabel('Retention Probability')
        ax.set_title('Signal Decay Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        ax.legend()
        plt.grid(True)
        plt.tight_layout()

        # Display numbers on top of the bars
        for bar in top_bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height*100:.1f}%', ha='center', va='bottom')
    
        for bar in bottom_bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height*100:.1f}%', ha='center', va='bottom')        
        plt.show()