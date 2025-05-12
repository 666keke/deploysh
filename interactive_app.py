import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import category_encoders as ce
import itertools # Not strictly used in final app but was in original
from scipy.stats import chi2_contingency

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="上海话数据交互分析平台")
st.title("📊 上海话问卷数据交互式分析平台")
st.markdown("""
欢迎使用上海话问卷数据交互分析平台！
请在左侧边栏选择筛选条件、分析指标和分组方式，右侧将展示相应的数据图表和表格。
""")

# --- Data Loading and Preprocessing (Adapted from share.py) ---
@st.cache_data # Cache the data loading and processing
def load_and_process_data():
    df = pd.read_csv('data.csv')

    # Initial filter as in share.py
    # Consider making this an optional filter in the UI if broader analysis is needed
    df = df[df['16.你是否愿意为了传承文化去特意学习上海话？'] == 'D.无所谓（请直接选择此项）'].copy() # Use .copy() to avoid SettingWithCopyWarning

    drop_cols = [
        '编号', '开始答题时间', '结束答题时间', '答题时长',
        '3.你来自于：_填空3',
        '语言', '清洗数据结果', '智能清洗数据无效概率',
        '地理位置国家和地区', '地理位置省', '地理位置市',
        '用户类型', '用户标识', '昵称', '自定义字段', 'IP', 'UA',
        'Referrer', '中奖时间', '中奖金额', '审核状态', 'Unnamed: 58',
        '16.你是否愿意为了传承文化去特意学习上海话？'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    multi_cols_21 = [c for c in df.columns if c.startswith('21.你通常在以下哪些场合使用上海话')]
    multi_cols_24 = [c for c in df.columns if c.startswith('24.你看过或听过以下哪类与上海话有关的内容')]
    for col in multi_cols_21 + multi_cols_24:
        df[col] = df[col].notna().astype(int)

    rename_map = {
        '1.你的性别是？': 'gender',
        '2.你的年级是？': 'grade',
        '4.你的专业类型是？': 'major',
        '5.你目前就读的学校是？':'school',
        '6.你是否为上海本地人？': 'native',
        '7.你的父母是否为上海本地人？':'parents',
        '8.你对上海话的整体印象是？': 'overall_impression',
        '9.你认为上海话属于一种：': 'shanghainese_attitude',
        '10.你认为学习/会说上海话是否是一种“本地身份”的象征？': 'identity',
        '14.你对在公共场合听到上海话的看法是？': 'public_hear_view',
        '15.你对私人场合使用上海话交流的看法是？': 'private_use_view',
        '11.你对大学中开设上海话课程的态度是？': 'course_attitude',
        '12.你是否认同“年轻一代应该会说一些上海话”？':'young_should',
        '13.你觉得上海话在现代社会中的地位是？':'social_status_of_shanghainese',
        '17.你是否觉得学校或社会应该提供更多学习上海话的机会？': 'more_learning_opportunity',
        '18.你是否会说上海话？': 'speaking_ability',
        '19.你能听懂上海话的程度是？': 'listening_ability',
        '20.你与家人交流时最常用的语言是？':'family_language',
        '21.你通常在以下哪些场合使用上海话？:与家人交流':'family_use',
        '21.你通常在以下哪些场合使用上海话？:与朋友交流':'friend_use',
        '21.你通常在以下哪些场合使用上海话？:在本地社区或邻里间':'local_community_use',
        '21.你通常在以下哪些场合使用上海话？:在工作/兼职中':'work_use',
        '21.你通常在以下哪些场合使用上海话？:基本不用':'basic_no_use',
        '22.你使用上海话的频率是？': 'usage_freq',
        '23.你会使用上海话发微信/社交平台信息吗？': 'sns_use_freq',
        '24.你看过或听过以下哪类与上海话有关的内容？:上海话配音短视频':'shanghainese_dubbling_tiktok',
        '24.你看过或听过以下哪类与上海话有关的内容？:上海话电视剧/电影':'shanghainese_movies',
        '24.你看过或听过以下哪类与上海话有关的内容？:上海话广播/音频节目':'shanghainese_radio',
        '24.你看过或听过以下哪类与上海话有关的内容？:上海话学习类内容':'shanghainese_learning_resources',
        '24.你看过或听过以下哪类与上海话有关的内容？:几乎没有接触过':'never_shanghainese_content',
        '25.你是否关注过沪语博主（如G僧东、册那队长等）？': 'follow_sh_blogger',
        '26.你对此类沪语视频的看法是？': 'video_view',
        '27.你是否曾因不会说上海话而感到尴尬/被排斥？': 'awkward_score',
        '28.你认为目前的语言环境是否支持上海话的使用？': 'env_support',
        '3.你来自于：_填空1': 'Province',
        '3.你来自于：_填空2': 'City',
    }
    df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)

    def clean_province(name):
        suffixes = ['省', '市', '自治区', '特别行政区', '回族', '壮族', '维吾尔', '维吾尔族']
        for suffix in suffixes:
            if isinstance(name, str) and name.endswith(suffix):
                return name[:-len(suffix)]
        return name

    province_to_region = {
        '上海': '上海本地', '江苏': '华东', '浙江': '华东', '安徽': '华东', '福建': '华东', '山东': '华东', '江西': '华东',
        '广东': '华南', '广西': '华南', '海南': '华南',
        '北京': '华北', '天津': '华北', '河北': '华北', '山西': '华北', '内蒙古': '华北',
        '四川': '西南', '重庆': '西南', '云南': '西南', '贵州': '西南', '西藏': '西南',
        '陕西': '西北', '甘肃': '西北', '青海': '西北', '宁夏': '西北', '新疆': '西北',
        '辽宁': '东北', '吉林': '东北', '黑龙江': '东北'
    }

    if 'City' in df.columns and 'Province' in df.columns:
        df.loc[df['City'].isin(['上海', '上海市']), 'Province'] = '上海'
        df.loc[df['City'].isin(['北京', '北京市']), 'Province'] = '北京'
        df.loc[df['City'].isin(['天津', '天津市']), 'Province'] = '天津'
        df.loc[df['City'].isin(['重庆', '重庆市']), 'Province'] = '重庆'

        df['Province'] = df['Province'].apply(clean_province)
        df['Province'] = df['Province'].apply(clean_province) # Apply twice as in original
        df['Region'] = df['Province'].map(province_to_region).fillna('其他地区')

    # Specific Encodings for analysis & creating string columns for filters
    gender_mapping_display = {1: '男', 0: '女', 2:'其他'} # For display
    if 'gender' in df.columns:
        df['gender_code'] = df['gender'].map({'A.男':1, 'B.女':0, 'C.其他':2})
        df['gender_str'] = df['gender_code'].map(gender_mapping_display)

    native_mapping_display = {'A.是，在上海出生并长大': '上海本地人(出生并长大)',
                              'B.否，但在上海生活超过5年': '长期居住上海(>5年)',
                              'C.否，在上海生活不足5年': '短期居住上海(<5年)'}
    if 'native' in df.columns:
        df['native_flag'] = (df['native'] == 'A.是，在上海出生并长大').astype(int)
        df['long_term_sh'] = (df['native'] == 'B.否，但在上海生活超过5年').astype(int)
        df['native_str'] = df['native'].map(lambda x: native_mapping_display.get(x, x))


    if 'follow_sh_blogger' in df.columns:
        df['follow_sh_blogger'] = df['follow_sh_blogger'].map({'A.是':1, 'B.否':0})
    if 'identity' in df.columns:
        df['identity'] = df['identity'].map({'A.是的':4, 'B.部分是':3, 'D.不清楚':2, 'C.否':1})
    if 'course_attitude' in df.columns:
        df['course_attitude'] = df['course_attitude'].map({'A.非常支持':4, 'B.支持':3, 'C.无所谓':2, 'D.反对':1})
    if 'young_should' in df.columns:
        df['young_should'] = df['young_should'].map({'A.非常认同':4, 'B.认同':3, 'C.不太认同':2, 'D.完全不认同':1})
    if 'shanghainese_attitude' in df.columns:
        df['shanghainese_attitude'] = df['shanghainese_attitude'].map({'A.地方语言，应予保护':4, 'B.沟通工具，实用即可':3, 'D.无所谓':2, 'C.方言，逐渐消失是自然现象':1})
    if 'social_status_of_shanghainese' in df.columns:
        df['social_status_of_shanghainese'] = df['social_status_of_shanghainese'].map({'A.重要，应重视':4, 'B.一般，可保留可取代':3, 'D.难说':2, 'C.不重要':1})
    if 'more_learning_opportunity' in df.columns:
        df['more_learning_opportunity'] = df['more_learning_opportunity'].map({'A.是':1, 'B.否':-1, 'C.无所谓': 0})
    if 'awkward_score' in df.columns:
        df['awkward_score'] = pd.to_numeric(df['awkward_score'], errors='coerce').fillna(0) # Ensure numeric before negation
        df['awkward_score'] = -df['awkward_score']

    original_major_col = df['major'].copy() if 'major' in df.columns else pd.Series()
    original_grade_col = df['grade'].copy() if 'grade' in df.columns else pd.Series()

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Columns to keep as strings for direct use (not to be ordinally encoded here if present)
    keep_as_string = ['Region', 'gender_str', 'native_str', 'Province', 'City', 'school', 'parents', 'family_language', 'gender', 'native']
    cat_cols_for_encoding = [col for col in cat_cols if col not in keep_as_string]

    major_mapping = {}
    grade_mapping = {}

    if cat_cols_for_encoding:
        encoder = ce.OrdinalEncoder(cols=cat_cols_for_encoding, handle_unknown='value', handle_missing='return_nan') # Use return_nan for explicit handling
        df = encoder.fit_transform(df)

        for mapping_info in encoder.category_mapping:
            col_name = mapping_info['col']
            current_map = {k:v for k,v in mapping_info['mapping'].items() if not (isinstance(k, float) and np.isnan(k))}
            if col_name == 'major':
                major_mapping = current_map
            elif col_name == 'grade':
                grade_mapping = current_map

    # Fallback or ensure major/grade mappings exist even if not in cat_cols_for_encoding (e.g. if already numeric by mistake)
    if not major_mapping and 'major' in df.columns:
        unique_majors = original_major_col.dropna().unique()
        major_mapping = {val: i+1 for i, val in enumerate(unique_majors)}
        df['major'] = original_major_col.map(major_mapping) # Re-map using original values if needed

    if not grade_mapping and 'grade' in df.columns:
        unique_grades = original_grade_col.dropna().unique()
        grade_mapping = {val: i+1 for i, val in enumerate(unique_grades)}
        df['grade'] = original_grade_col.map(grade_mapping) # Re-map

    # Define attitude_cols (inferred and explicit)
    attitude_cols = [
        'overall_impression', 'shanghainese_attitude', 'identity',
        'public_hear_view', 'private_use_view', 'course_attitude',
        'young_should', 'social_status_of_shanghainese',
        'more_learning_opportunity', 'video_view', 'awkward_score', 'env_support'
    ]
    # Filter out cols not in df from attitude_cols
    attitude_cols = [col for col in attitude_cols if col in df.columns]

    question_cols_list = attitude_cols + [
        'speaking_ability', 'listening_ability', 'usage_freq', 'sns_use_freq',
        'family_use', 'friend_use', 'local_community_use', 'work_use', 'basic_no_use',
        'shanghainese_dubbling_tiktok', 'shanghainese_movies', 'shanghainese_radio',
        'shanghainese_learning_resources', 'never_shanghainese_content',
        'follow_sh_blogger'
    ]

    final_question_cols = []
    for col in question_cols_list:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Drop rows where essential numeric question columns became NaN after conversion, or fill them
            # df.dropna(subset=[col], inplace=True) # This might reduce data significantly
            # For now, we let mean handle NaNs, but this is a point of attention for data quality.
            final_question_cols.append(col)
    final_question_cols = sorted(list(set(final_question_cols))) # Unique and sorted

    # Create string versions for major and grade for grouping display using the mappings
    # Mappings are {original_name: encoded_code}
    rev_major_mapping = {v: k for k, v in major_mapping.items()} if major_mapping else {}
    rev_grade_mapping = {v: k for k, v in grade_mapping.items()} if grade_mapping else {}

    if 'major' in df.columns and major_mapping:
      df['major_str'] = df['major'].map(rev_major_mapping).fillna('未知')
    if 'grade' in df.columns and grade_mapping:
      df['grade_str'] = df['grade'].map(rev_grade_mapping).fillna('未知')

    return df, final_question_cols, rename_map, major_mapping, grade_mapping, gender_mapping_display, native_mapping_display, province_to_region

df_processed, question_cols, rename_map, major_mapping, grade_mapping, gender_map_disp, native_map_disp, prov_to_region_map = load_and_process_data()

# --- Sidebar for Controls ---
st.sidebar.header("⚙️ 筛选与可视化选项")

# Metrics and Grouping Selection
rev_rename_map = {v: k for k, v in rename_map.items()}
def get_display_name(q_col):
    original_question = rev_rename_map.get(q_col, q_col)
    # Simplify common question formats
    name = original_question.split("？:")[-1].split('？')[-1].split('是：')[-1]
    name = name.replace('（请直接选择此项）','').replace("（", "(").replace("）", ")").strip()
    if len(name) > 50: # Truncate very long names
        name = name[:47] + "..."
    return name if name else q_col

question_cols_display_names = {}
if question_cols:
    question_cols_display_names = {q_col: q_col for q_col in question_cols}

# Initialize threshold analysis variables
threshold_metric = None
threshold_values = []

# Create filter options
major_filter_options = []
if major_mapping:
    major_filter_options = sorted([k for k in major_mapping.keys() if not (isinstance(k, float) and np.isnan(k))])
selected_major_names = st.sidebar.multiselect("🎓 选择专业类型", options=major_filter_options, default=major_filter_options)
selected_major_codes = [major_mapping[name] for name in selected_major_names if name in major_mapping]

grade_filter_options = []
if grade_mapping:
    grade_filter_options = sorted([k for k in grade_mapping.keys() if not (isinstance(k, float) and np.isnan(k))])
selected_grade_names = st.sidebar.multiselect("📈 选择年级", options=grade_filter_options, default=grade_filter_options)
selected_grade_codes = [grade_mapping[name] for name in selected_grade_names if name in grade_mapping]

region_options = []
if 'Region' in df_processed.columns:
    region_options = sorted(df_processed['Region'].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect("🗺️ 选择地区", options=region_options, default=region_options)

gender_str_options = []
if 'gender_str' in df_processed.columns:
    gender_str_options = sorted(df_processed['gender_str'].dropna().unique().tolist())
selected_genders_str = st.sidebar.multiselect("🚻 选择性别", options=gender_str_options, default=gender_str_options)

native_str_options = []
if 'native_str' in df_processed.columns:
    native_str_options = sorted(df_processed['native_str'].dropna().unique().tolist())
selected_natives_str = st.sidebar.multiselect("🏠 选择上海人身份", options=native_str_options, default=native_str_options)

# Apply filters
filtered_df = df_processed.copy()
if selected_major_codes and 'major' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['major'].isin(selected_major_codes)]
if selected_grade_codes and 'grade' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['grade'].isin(selected_grade_codes)]
if selected_regions and 'Region' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Region'].isin(selected_regions)]
if selected_genders_str and 'gender_str' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['gender_str'].isin(selected_genders_str)]
if selected_natives_str and 'native_str' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['native_str'].isin(selected_natives_str)]


# Metrics and Grouping Selection

selected_metrics_keys = st.sidebar.multiselect(
    "📊 选择分析指标 (Y轴)",
    options=list(question_cols_display_names.keys()),
    format_func=lambda x: question_cols_display_names[x],
    default=[question_cols[0]] if question_cols else []
)

# Grouping variables (use string versions for display)
grouping_options_map = {}
if 'Region' in filtered_df.columns:
    grouping_options_map['Region'] = "地区"
if 'gender_str' in filtered_df.columns:
    grouping_options_map['gender_str'] = "性别"
if 'native_str' in filtered_df.columns:
    grouping_options_map['native_str'] = "上海人身份"
if 'major_str' in filtered_df.columns and filtered_df['major_str'].nunique() > 0 :
    grouping_options_map['major_str'] = "专业类型"
if 'grade_str' in filtered_df.columns and filtered_df['grade_str'].nunique() > 0:
    grouping_options_map['grade_str'] = "年级"


if grouping_options_map:
    selected_group_by_key = st.sidebar.selectbox(
        "🗂️ 选择分组条件 (X轴)",
        options=list(grouping_options_map.keys()),
        format_func=lambda x: grouping_options_map[x],
        index=0
    )
else:
    st.sidebar.warning("⚠️ 没有可用的分组条件。请检查数据。")
    selected_group_by_key = None


# --- Main Area for Charts and Tables ---
if not selected_metrics_keys or not selected_group_by_key or filtered_df.empty:
    st.warning("⚠️ 请至少选择一个分析指标和一个分组条件，并确保筛选结果不为空。")
    if filtered_df.empty:
        st.info("当前筛选条件下没有数据。请尝试调整筛选器。")
else:
    st.subheader("📈 图表分析")

    for metric_key in selected_metrics_keys:
        metric_display_name = question_cols_display_names.get(metric_key, metric_key)
        group_by_display_name = grouping_options_map.get(selected_group_by_key, selected_group_by_key)

        if metric_key not in filtered_df.columns:
            st.error(f"指标 '{metric_display_name}' ({metric_key}) 在筛选后的数据中不存在。")
            continue
        if selected_group_by_key not in filtered_df.columns:
            st.error(f"分组条件 '{group_by_display_name}' ({selected_group_by_key}) 在筛选后的数据中不存在。")
            continue

        try:
            if not pd.api.types.is_numeric_dtype(filtered_df[metric_key]):
                 st.error(f"指标 '{metric_display_name}' ({metric_key}) 不是数值类型，无法计算均值。")
                 continue

            # Drop NA for the specific metric and group_by col before grouping to avoid errors with all-NA groups
            temp_plot_df = filtered_df[[selected_group_by_key, metric_key]].dropna(subset=[metric_key, selected_group_by_key])
            if temp_plot_df.empty:
                st.info(f"指标 '{metric_display_name}' 按 '{group_by_display_name}' 分组后无有效数据可供绘图。")
                continue

            plot_df = temp_plot_df.groupby(selected_group_by_key, as_index=False)[metric_key].mean()
            plot_df = plot_df.sort_values(by=metric_key, ascending=False)

            if plot_df.empty:
                st.info(f"指标 '{metric_display_name}' 按 '{group_by_display_name}' 分组后无数据可供绘图。")
                continue

            fig_title = f"'{metric_display_name}' 按 '{group_by_display_name}' 分布 (均值)"
            fig = px.bar(plot_df, x=selected_group_by_key, y=metric_key,
                         title=fig_title,
                         labels={metric_key: f"均值 - {metric_display_name}", selected_group_by_key: group_by_display_name},
                         color=selected_group_by_key,
                         text_auto='.2f')
            fig.update_layout(
                xaxis_title=group_by_display_name,
                yaxis_title=f"均值 - {metric_display_name}",
                title_x=0.5,
                legend_title_text=group_by_display_name
            )
            st.plotly_chart(fig, use_container_width=True)

            csv_fig_data = plot_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label=f"📥 下载图表 '{metric_display_name}' 数据 (CSV)",
                data=csv_fig_data,
                file_name=f"{metric_key}_by_{selected_group_by_key}.csv",
                mime='text/csv',
                key=f"download_chart_{metric_key}_{selected_group_by_key}"
            )
            st.markdown("---")

        except Exception as e:
            st.error(f"为指标 '{metric_display_name}' 和分组 '{group_by_display_name}' 生成图表时出错: {e}")


    st.subheader("📄 筛选后数据预览 (前100条)")
    display_cols = []
    if selected_group_by_key:
        display_cols.append(selected_group_by_key)
    display_cols.extend(selected_metrics_keys)
    display_cols.extend([col for col in ['major_str', 'grade_str', 'Region', 'gender_str', 'native_str']
                    if col != selected_group_by_key and col in filtered_df.columns])
    st.dataframe(filtered_df[list(dict.fromkeys(display_cols))].head(100))

    csv_filtered_data = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下载筛选后完整数据 (CSV)",
        data=csv_filtered_data,
        file_name="filtered_shanghainese_data.csv",
        mime='text/csv',
        key="download_filtered_all"
    )

    # --- Threshold Analysis Section ---
    if not selected_metrics_keys or not selected_group_by_key or filtered_df.empty:
        pass  # Don't show threshold analysis if no metrics or filtered data
    else:
        st.markdown("---")
        st.subheader("📊 阈值分析")

        enable_threshold_analysis = st.checkbox("启用阈值分析", value=False)

        if enable_threshold_analysis:
            if not selected_metrics_keys:
                st.warning("请先在侧边栏选择至少一个分析指标 (Y轴)。")
            else:
                # Use the first selected metric for threshold analysis
                threshold_metric = selected_metrics_keys[0]
                metric_display_name = question_cols_display_names.get(threshold_metric, threshold_metric)

                st.write(f"当前分析指标: **{metric_display_name}**")

                threshold_input = st.text_input(
                    "输入阈值（用逗号分隔，例如：1,2,3）",
                    key="threshold_input"
                )

                if threshold_input:
                    try:
                        threshold_values = [float(x.strip()) for x in threshold_input.split(",") if x.strip()]
                        threshold_values.sort()
                    except ValueError:
                        st.error("请输入有效的数字，用逗号分隔")

                if not threshold_values:
                    st.warning("请输入至少一个阈值。")
                elif threshold_metric not in filtered_df.columns:
                    st.error(f"所选指标 '{metric_display_name}' 在筛选后的数据中不存在。")
                else:
                    # Create interval labels
                    interval_labels = []
                    for i in range(len(threshold_values) + 1):
                        if i == 0:
                            interval_labels.append(f"< {threshold_values[0]}")
                        elif i == len(threshold_values):
                            interval_labels.append(f">= {threshold_values[-1]}")
                        else:
                            interval_labels.append(f"{threshold_values[i-1]} - {threshold_values[i]}")

                    # Calculate counts for each interval
                    interval_counts = []
                    valid_data = filtered_df[pd.notna(filtered_df[threshold_metric])]

                    for i in range(len(threshold_values) + 1):
                        if i == 0:
                            count = sum(valid_data[threshold_metric] < threshold_values[0])
                        elif i == len(threshold_values):
                            count = sum(valid_data[threshold_metric] >= threshold_values[-1])
                        else:
                            count = sum((valid_data[threshold_metric] >= threshold_values[i-1]) &
                                         (valid_data[threshold_metric] < threshold_values[i]))
                        interval_counts.append(count)

                    # Create DataFrame for visualization
                    total_count = sum(interval_counts)
                    percentages = [count / total_count * 100 if total_count > 0 else 0 for count in interval_counts]

                    threshold_df = pd.DataFrame({
                        '区间': interval_labels,
                        '人数': interval_counts,
                        '百分比 (%)': [f"{p:.2f}%" for p in percentages]
                    })

                    # Create two columns for charts
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("柱状图")
                        fig_bar = px.bar(
                            threshold_df,
                            x='区间',
                            y='人数',
                            title=f"'{metric_display_name}' 的阈值分析",
                            text='人数'
                        )
                        fig_bar.update_layout(
                            xaxis_title="分数区间",
                            yaxis_title="人数",
                            title_x=0.5
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    with col2:
                        st.subheader("饼图")
                        fig_pie = px.pie(
                            threshold_df,
                            values='人数',
                            names='区间',
                            title=f"'{metric_display_name}' 的区间分布",
                            hover_data=['百分比 (%)']
                        )
                        fig_pie.update_layout(
                            title_x=0.5
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    # Display data table
                    st.subheader("区间人数统计表")
                    st.dataframe(threshold_df)

                    # Display histogram
                    st.subheader("数据分布直方图")

                    # Create histogram with threshold lines
                    fig_hist = px.histogram(
                        valid_data,
                        x=threshold_metric,
                        nbins=20,
                        title=f"'{metric_display_name}' 的分布直方图",
                        labels={threshold_metric: metric_display_name}
                    )

                    # Add vertical lines for thresholds
                    for threshold in threshold_values:
                        fig_hist.add_vline(
                            x=threshold,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"阈值: {threshold}",
                            annotation_position="top right"
                        )

                    fig_hist.update_layout(
                        xaxis_title=metric_display_name,
                        yaxis_title="频数",
                        title_x=0.5
                    )

                    st.plotly_chart(fig_hist, use_container_width=True)

                    # Display descriptive statistics
                    st.subheader("描述性统计")

                    # Calculate descriptive statistics
                    desc_stats = valid_data[threshold_metric].describe()

                    # Create a DataFrame for display
                    stats_df = pd.DataFrame({
                        '统计量': ['样本数', '平均值', '标准差', '最小值', '25%分位数', '中位数', '75%分位数', '最大值'],
                        '值': [
                            f"{desc_stats['count']:.0f}",
                            f"{desc_stats['mean']:.2f}",
                            f"{desc_stats['std']:.2f}",
                            f"{desc_stats['min']:.2f}",
                            f"{desc_stats['25%']:.2f}",
                            f"{desc_stats['50%']:.2f}",
                            f"{desc_stats['75%']:.2f}",
                            f"{desc_stats['max']:.2f}"
                        ]
                    })

                    st.dataframe(stats_df)

                    # Add download buttons for data
                    col1, col2 = st.columns(2)

                    with col1:
                        # Add download button for threshold analysis data
                        csv_threshold_data = threshold_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label=f"📥 下载区间统计数据 (CSV)",
                            data=csv_threshold_data,
                            file_name=f"{threshold_metric}_threshold_analysis.csv",
                            mime='text/csv',
                            key="download_threshold_analysis"
                        )

                    with col2:
                        # Add download button for descriptive statistics
                        csv_stats_data = stats_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label=f"📥 下载描述性统计数据 (CSV)",
                            data=csv_stats_data,
                            file_name=f"{threshold_metric}_descriptive_stats.csv",
                            mime='text/csv',
                            key="download_descriptive_stats"
                        )

                    # --- Category Table Analysis ---
                    st.markdown("---")
                    st.subheader("📊 类别表格分析")

                    enable_category_analysis = st.checkbox("启用类别表格分析", value=False)
                    show_totals = st.checkbox("显示总计", value=True)

                    if enable_category_analysis:
                        # Select grouping variable for categories
                        category_options = []
                        if 'Region' in filtered_df.columns:
                            category_options.append(('Region', "地区"))
                        if 'gender_str' in filtered_df.columns:
                            category_options.append(('gender_str', "性别"))
                        if 'native_str' in filtered_df.columns:
                            category_options.append(('native_str', "上海人身份"))
                        if 'major_str' in filtered_df.columns and filtered_df['major_str'].nunique() > 0:
                            category_options.append(('major_str', "专业类型"))
                        if 'grade_str' in filtered_df.columns and filtered_df['grade_str'].nunique() > 0:
                            category_options.append(('grade_str', "年级"))

                        if not category_options:
                            st.warning("没有可用的分类变量。")
                        else:
                            # Create a dictionary for the selectbox format_func
                            category_options_dict = {k: v for k, v in category_options}

                            selected_category = st.selectbox(
                                "选择类别变量",
                                options=[k for k, _ in category_options],
                                format_func=lambda x: category_options_dict.get(x, x),
                                key="category_select"
                            )

                            if selected_category:
                                # Get unique categories
                                categories = sorted(valid_data[selected_category].dropna().unique())

                                if len(categories) > 0:
                                    # Allow user to select which categories to include
                                    selected_categories = st.multiselect(
                                        "选择要包含的类别",
                                        options=categories,
                                        default=categories,
                                        key="selected_categories"
                                    )

                                    # Add custom group functionality
                                    st.subheader("自定义聚合群体")
                                    enable_custom_groups = st.checkbox("启用自定义聚合群体", value=False)

                                    custom_groups = {}
                                    if enable_custom_groups:
                                        st.write("创建自定义群体（将多个类别聚合为一个群体）")

                                        # UI for creating custom groups
                                        col1, col2 = st.columns([1, 2])
                                        with col1:
                                            custom_group_name = st.text_input("群体名称", key="custom_group_name")
                                        with col2:
                                            group_categories = st.multiselect(
                                                "选择要聚合的类别",
                                                options=categories,
                                                key="group_categories"
                                            )

                                        if st.button("添加自定义群体", key="add_custom_group"):
                                            if custom_group_name and group_categories:
                                                custom_groups[custom_group_name] = group_categories
                                                st.success(f"已添加自定义群体: {custom_group_name}")

                                        # Display current custom groups
                                        if 'custom_groups' in st.session_state:
                                            for name, cats in st.session_state.custom_groups.items():
                                                st.write(f"- {name}: {', '.join(cats)}")

                                        # Store custom groups in session state
                                        if custom_groups:
                                            if 'custom_groups' not in st.session_state:
                                                st.session_state.custom_groups = {}
                                            st.session_state.custom_groups.update(custom_groups)

                                    # Get custom groups from session state
                                    if 'custom_groups' in st.session_state:
                                        custom_groups = st.session_state.custom_groups

                                    if selected_categories or custom_groups:
                                        # Create interval labels (same as before)
                                        interval_labels = []
                                        for i in range(len(threshold_values) + 1):
                                            if i == 0:
                                                interval_labels.append(f"< {threshold_values[0]}")
                                            elif i == len(threshold_values):
                                                interval_labels.append(f">= {threshold_values[-1]}")
                                            else:
                                                interval_labels.append(f"{threshold_values[i-1]} - {threshold_values[i]}")

                                        # Get all categories that need to be included (selected categories + all categories in any custom group)
                                        categories_to_include = set(selected_categories) if selected_categories else set()
                                        for group_name, group_cats in custom_groups.items():
                                            categories_to_include.update(group_cats)

                                        # Filter data to include all necessary categories
                                        category_data = valid_data[valid_data[selected_category].isin(categories_to_include)] if categories_to_include else valid_data

                                        # Combine selected categories with custom groups
                                        all_categories = list(selected_categories) if selected_categories else []
                                        for group_name in custom_groups:
                                            if group_name not in all_categories:
                                                all_categories.append(group_name)

                                        # Create contingency table with categories as rows and intervals as columns
                                        contingency_table = np.zeros((len(all_categories), len(interval_labels)))

                                        # Fill contingency table
                                        for i, category in enumerate(all_categories):
                                            # Check if this is a custom group
                                            if category in custom_groups:
                                                # Combine data from multiple categories
                                                group_categories = custom_groups[category]
                                                category_subset = category_data[category_data[selected_category].isin(group_categories)]
                                            else:
                                                # Regular category
                                                category_subset = category_data[category_data[selected_category] == category]

                                            for j, interval in enumerate(interval_labels):
                                                if j == 0:  # First interval
                                                    count = sum(category_subset[threshold_metric] < threshold_values[0])
                                                elif j == len(threshold_values):  # Last interval
                                                    count = sum(category_subset[threshold_metric] >= threshold_values[-1])
                                                else:  # Middle intervals
                                                    count = sum((category_subset[threshold_metric] >= threshold_values[j-1]) &
                                                                (category_subset[threshold_metric] < threshold_values[j]))

                                                contingency_table[i, j] = count

                                        # Calculate row totals (for categories)
                                        category_totals = np.sum(contingency_table, axis=1)

                                        # Calculate column totals (for intervals)
                                        interval_totals = np.sum(contingency_table, axis=0)

                                        # Perform chi-square test if we have enough data
                                        if np.sum(contingency_table) > 0 and np.all(category_totals > 0) and np.all(interval_totals > 0):
                                            chi2, p, dof, expected = chi2_contingency(contingency_table)
                                            chi2_result = f"χ² = {chi2:.2f}, p = {p:.4f}"
                                        else:
                                            chi2_result = "数据不足，无法进行卡方检验"

                                        # Create DataFrame for display with categories as rows and intervals as columns
                                        category_table = pd.DataFrame(contingency_table, index=all_categories, columns=interval_labels)

                                        # Add row totals (for categories) if show_totals is checked
                                        if show_totals:
                                            category_table['总计'] = category_totals

                                            # Add column totals (for intervals)
                                            category_table.loc['总计'] = list(interval_totals) + [np.sum(contingency_table)]

                                        # Always add chi-square test results regardless of show_totals
                                        # Ensure the chi-square test row has the same number of columns as the table
                                        empty_values = [''] * (len(category_table.columns) - 1)
                                        category_table.loc['卡方检验'] = [chi2_result] + empty_values

                                        # Display the table
                                        st.subheader(f"按 {category_options_dict.get(selected_category, selected_category)} 分类的区间统计表")
                                        st.dataframe(category_table)

                                        # Add download button for category table
                                        csv_category_table = category_table.to_csv(index=True).encode('utf-8-sig')
                                        st.download_button(
                                            label=f"📥 下载类别表格数据 (CSV)",
                                            data=csv_category_table,
                                            file_name=f"{threshold_metric}_category_analysis.csv",
                                            mime='text/csv',
                                            key="download_category_analysis"
                                        )
                                    else:
                                        st.warning("请选择至少一个类别或创建自定义聚合群体。")
                                else:
                                    st.warning(f"选定的类别变量 '{category_options_dict.get(selected_category, selected_category)}' 没有有效数据。")

# --- Encoding Information ---
with st.expander("ℹ️ 查看编码说明和原始问卷信息"):
    st.markdown("#### **问卷问题与编码后变量名映射**")
    rename_df_display = pd.DataFrame(list(rename_map.items()), columns=['原始问卷问题', '编码后变量名'])
    st.table(rename_df_display)

    st.markdown("#### **具体编码详情**")

    st.markdown("**性别 (gender_str / gender_code):**")
    gender_df_disp = pd.DataFrame({'原问卷答案': ['A.男', 'B.女'], '编码值 (gender_code)': [1,0], '分析用标签 (gender_str)': ['男','女']})
    st.table(gender_df_disp)

    st.markdown("**上海人身份 (native_str):**")
    # native_mapping_display from load_and_process_data
    native_df_disp = pd.DataFrame(list(native_map_disp.items()), columns=['原问卷答案', '分析用标签'])
    st.table(native_df_disp)

    st.markdown("**专业类型 (major_str / major):**")
    if major_mapping:
        major_df_disp = pd.DataFrame(list(major_mapping.items()), columns=['原问卷答案', '编码值 (major)'])
        st.table(major_df_disp.sort_values(by='编码值 (major)'))
    else:
        st.markdown("_无专业类型数据或映射_")

    st.markdown("**年级 (grade_str / grade):**")
    if grade_mapping:
        grade_df_disp = pd.DataFrame(list(grade_mapping.items()), columns=['原问卷答案', '编码值 (grade)'])
        st.table(grade_df_disp.sort_values(by='编码值 (grade)'))
    else:
        st.markdown("_无年级数据或映射_")

    course_attitude_map_disp = {'A.非常支持':4, 'B.支持':3, 'C.无所谓':2, 'D.反对':1}
    st.markdown("**大学开设上海话课程态度 (course_attitude):**")
    st.table(pd.DataFrame(list(course_attitude_map_disp.items()), columns=['原问卷答案', '编码值']))

    identity_map_disp = {'A.是的':4, 'B.部分是':3, 'D.不清楚':2, 'C.否':1}
    st.markdown("**学习/会说上海话是‘本地身份’象征 (identity):**")
    st.table(pd.DataFrame(list(identity_map_disp.items()), columns=['原问卷答案', '编码值']))

    st.markdown("**省份与地区对应 (Region):**")
    st.table(pd.DataFrame(list(prov_to_region_map.items()), columns=['省份', '地区']))


st.sidebar.markdown("---")
st.sidebar.info("""
**💡 运行指南:**
1. 确保 `data.csv` 文件与此 `interactive_app.py` 在同一目录下。
2. 安装必要的库: 
   `pip install streamlit pandas plotly openpyxl category_encoders`
3. 在终端中运行: 
   `streamlit run interactive_app.py`
""")


st.markdown("---")
st.markdown("Shanghainese Dialect Survey Interactive Analysis | Developed with Streamlit") 
