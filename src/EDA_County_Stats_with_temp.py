import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.optimize import least_squares
from sklearn.model_selection import train_test_split


df = pd.read_excel('County_Statistics_with_Temp.xlsx')
df_Emergency = df.dropna(subset=['Emergency Visits / 100000'])
df_Emergency = df_Emergency.drop('Hospitalizations / 100000', axis=1)


def histogram(feature, color):
    sns.histplot(df[feature], bins=15, kde=True, color=color)
    plt.title(f'Distribution of {feature} (All Counties)')
    plt.xlabel(feature)
    plt.ylabel('Number of Counties')
    plt.show()


# Notes: A few counties have high ER visit and that corresponds to high heat counties like Imperial County
# Same with hospitalization
# I am considering a piecewise relationship, since there might be a threshold temperature
# Below threshold temperature, no unusual risk of heat stroke
# https://www.dir.ca.gov/dosh/heatillnessinvestigations-2006.pdf
# https://www.lni.wa.gov/safety-health/safety-training-materials/workshops-events/beheatsmart#questions-and-answers
# Both of these suggest to set the threshold to 80 F


def fit_linear_with_intercept_enforced(x, y, lower, upper,title, x_thresh=80.0, y_min=None,
                                       plot=True, max_nfev=20000):
    """
    Fit y = m * x + b to points with x >= x_thresh, enforcing b so that y(x_thresh) = y_min.
    y_min defaults to the min of the original y array (ignoring NaNs) if not provided.

    Args:
        x, y: 1-D array-like (will be converted to numpy arrays). NaNs are handled.
        lower, upper: bounds for parameter vector [m]. Each must be scalar or length-1 sequence.
        x_thresh: threshold where intercept is enforced (default 80.0).
        y_min: value y(x_thresh) should equal. If None, uses np.nanmin(y) from the original y.
        plot: if True, show a scatter + fitted line plot.
        max_nfev: max evaluations for least_squares.

    Returns:
        dict with keys: m, b, success, message, rmse, r2, res (least_squares result), x_fit_mask, x, y, y_pred
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # determine y_min if not provided (use full original y array)
    if y_min is None:
        y_min = np.nanmin(y_arr)

    # mask for x >= threshold and non-NaN pairs
    mask = (x_arr >= x_thresh) & ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_sel = x_arr[mask]
    y_sel = y_arr[mask]

    if x_sel.size == 0:
        raise ValueError(f"No data points with x >= {x_thresh}")

    # model: y = m * x + b with b = y_min - m * x_thresh
    def model_given_m(params, xvals):
        m = params[0]
        b = y_min - m * x_thresh
        return m * xvals + b

    def residuals(params, xvals, yvals):
        return model_given_m(params, xvals) - yvals

    # initial guess for m: slope from simple linear fit on selected points if possible
    try:
        slope_init = np.polyfit(x_sel, y_sel, 1)[0]
        m_init = slope_init
    except Exception:
        m_init = (np.nanmean(y_sel) / np.nanmean(x_sel)) if np.nanmean(x_sel) != 0 else 1.0

    init = np.atleast_1d(np.array([m_init], dtype=float))

    # ensure bounds are arrays of length 1
    lower_arr = np.atleast_1d(np.array(lower, dtype=float))
    upper_arr = np.atleast_1d(np.array(upper, dtype=float))
    if lower_arr.size != 1 or upper_arr.size != 1:
        raise ValueError("lower and upper must be scalar or length-1 sequences for parameter [m].")

    res = least_squares(residuals, init, args=(x_sel, y_sel), bounds=(lower_arr, upper_arr),
                        ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=max_nfev)

    m_fit = res.x[0]
    b_fit = y_min - m_fit * x_thresh

    # diagnostics on selected points
    y_pred_sel = model_given_m([m_fit], x_sel)
    rss = np.sum((y_sel - y_pred_sel) ** 2)
    rmse = np.sqrt(rss / len(y_sel))
    r2 = 1 - rss / np.sum((y_sel - np.mean(y_sel)) ** 2) if np.sum((y_sel - np.mean(y_sel)) ** 2) != 0 else np.nan

    # optional plot (shows only selected points and fitted line)
    if plot:
        xs = np.linspace(np.nanmin(x_sel) - 1, np.nanmax(x_sel) + 1, 400)
        ys = m_fit * xs + b_fit
        plt.figure(figsize=(8,6))
        plt.scatter(x_sel, y_sel, label='data (x >= {:.1f})'.format(x_thresh), c='C0')
        plt.plot(xs, ys, 'r-', lw=2, label=f'fit: y = {m_fit:.4g}*x + {b_fit:.4g}')
        plt.scatter([x_thresh], [y_min], color='black', zorder=5, label=f'y({x_thresh}) enforced = {y_min:.3g}')
        plt.xlabel(f'x (>= {x_thresh})')
        plt.ylabel('y')
        plt.legend()
        plt.title('Linear fit for '+title)
        plt.show()

    return {
        'm': m_fit,
        'b': b_fit,
        'success': res.success,
        'message': res.message,
        'rmse': rmse,
        'r2': r2,
        'res': res,
        'x_fit_mask': mask,
        'x_selected': x_sel,
        'y_selected': y_sel,
        'y_pred_selected': y_pred_sel
    }


def temperature_plots():
    x_july = df_Emergency['July max temp (F)']
    x_august = df_Emergency['August max temp (F)']
    # as numpy arrays
    x_july_np = x_july.to_numpy(dtype=float)
    x_august_np = x_august.to_numpy(dtype=float)
    # elementwise max: np.fmax returns the non-NaN value when one side is NaN
    x_max_np = np.fmax(x_july_np, x_august_np)

    y = df_Emergency['Emergency Visits / 100000']
    y_np = y.to_numpy()
    y_list = y.tolist()

    lower = [0]
    upper = [1e6]

    out = fit_linear_with_intercept_enforced(x_july_np, y_list, lower, upper, title = 'July',
                                             x_thresh=80.0, y_min=np.nanmin(y_list), plot=True)
    print("m =", out['m'], "b =", out['b'], "RMSE =", out['rmse'], "R2 =", out['r2'])

    out = fit_linear_with_intercept_enforced(x_august_np, y_list, lower, upper, title = 'August',
                                             x_thresh=80.0, y_min=np.nanmin(y_list), plot=True)
    print("m =", out['m'], "b =", out['b'], "RMSE =", out['rmse'], "R2 =", out['r2'])


    out = fit_linear_with_intercept_enforced(x_max_np, y_list, lower, upper, title = 'max temp',
                                             x_thresh=80.0, y_min=np.nanmin(y_list), plot=True)
    print("m =", out['m'], "b =", out['b'], "RMSE =", out['rmse'], "R2 =", out['r2'])


temperature_plots()
# Ok, now I have an acceptable relationship between temperature and emergency visit.
# August has a lower
# Now I need to normalize this
# We can use this relationship, and subtract it, to get our residual plot
# Time to redo EDA, but this time with residuals instead


sns.set_style("whitegrid")
county_df = pd.read_excel("County_Statistics_with_Temp.xlsx")
# dropping hospitalization
county_df = county_df.drop('Hospitalizations / 100000', axis=1)
# drop na to drop rows with empty values for emergency visits
# our target is emergency visits for this analysis

county_df = county_df.dropna()
county_df.head()
# remove all instances of '%'
county_df['Park within 1/2 Mile'] = county_df['Park within 1/2 Mile'].str.removesuffix('%')
county_df['Imperviousness'] = county_df['Imperviousness'].str.removesuffix('%')
county_df['Energy Burden % of Income'] = county_df['Energy Burden % of Income'].str.removesuffix('%')
county_df['Housing Built before 1980'] = county_df['Housing Built before 1980'].str.removesuffix('%')
county_df['Housing Insecurity'] = county_df['Housing Insecurity'].str.removesuffix('%')
county_df['Lack of Reliable Transportation'] = county_df['Lack of Reliable Transportation'].str.removesuffix('%')
county_df['% w/o Internet'] = county_df['% w/o Internet'].str.removesuffix('%')
county_df['Utility Services Threat'] = county_df['Utility Services Threat'].str.removesuffix('%')

# convert to float
county_df['Park within 1/2 Mile'] = county_df['Park within 1/2 Mile'].astype(float)
county_df['Imperviousness'] = county_df['Imperviousness'].astype(float)
county_df['Energy Burden % of Income'] = county_df['Energy Burden % of Income'].astype(float)
county_df['Hospital Beds / 10000'] = county_df['Hospital Beds / 10000'].astype(float)
county_df['Housing Built before 1980'] = county_df['Housing Built before 1980'].astype(float)
county_df['Housing Insecurity'] = county_df['Housing Insecurity'].astype(float)
county_df['Lack of Reliable Transportation'] = county_df['Lack of Reliable Transportation'].astype(float)
county_df['% w/o Internet'] = county_df['% w/o Internet'].astype(float)
county_df['Utility Services Threat'] = county_df['Utility Services Threat'].astype(float)


# copy to avoid modifying original unless intended
county_df_residual = county_df.copy()

# column names
july_col = "July max temp (F)"
august_col = "August max temp (F)"
y_col = "Emergency Visits / 100000"

# compute elementwise max of July/August (np.fmax returns non-NaN if one side is NaN)
county_df_residual['max_july_august_temp'] = np.fmax(county_df_residual[july_col].astype(float).to_numpy(),
                                            county_df_residual[august_col].astype(float).to_numpy())

# fitted parameters (from your fit)
m = 1.8291274417820222
b = -143.4401953425618   # set so that y(80)=y_min in your fit
# compute y_hat and residuals using the max temp
temp = county_df_residual['max_july_august_temp'].astype(float)
y_obs = county_df_residual[y_col].astype(float)

# y_hat will be NaN wherever temp or y_obs is NaN (we keep NaNs)
y_hat = m * temp + b
y_resid = y_obs - y_hat

# assign new columns
resid_col = y_col + " temp residual"
county_df_residual[resid_col] = y_resid
county_df_residual = county_df_residual[county_df_residual['max_july_august_temp'] >= 80]


def pairplot_residual(features, imperial=False):
    if imperial:
        residual = county_df_residual

    else:
        residual = county_df_residual[county_df_residual['County'] != "Imperial"]

    county_train, county_test = train_test_split(
        county_df_residual, test_size=0.2, random_state=216, shuffle=True
    )

    sns.pairplot(county_train,
        y_vars=["Emergency Visits / 100000 temp residual"],
        x_vars=features,
        height=5,
        diag_kind=None,
    )

    plt.show()

    print("Correlations with Emergency Visits / 100000 residuals")
    print(county_df_residual[features].corrwith(county_df_residual['Emergency Visits / 100000 temp residual']))
    print()

    sns.pairplot(data=county_train,
                 x_vars=features,
                 y_vars=features,
                 hue=county_train.columns[0],
                 plot_kws={'alpha': .6})

    return county_train, county_test


pairplot_residual(county_df_residual.columns[2:11], imperial=True)

# We are going to use 4 features
features = ['Energy Burden % of Income', 'Park within 1/2 Mile', 'Imperviousness', '% w/o Internet']
county_train, county_test = pairplot_residual(features)

clean_county_train = county_train[['County', *features, 'Emergency Visits / 100000 temp residual']]
clean_county_validate = county_test[['County', *features, 'Emergency Visits / 100000 temp residual']]

clean_county_train.to_csv('src/train_post_EDA.csv', index=False)
clean_county_validate.to_csv('src/validate_post_EDA.csv', index=False)
