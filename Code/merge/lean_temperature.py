import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import pylab as P
import matplotlib.pyplot as plt
from scipy import optimize
import requests
import textwrap as tw
import time
import geocoder
import json
from vincenty import vincenty

homedir = os.getcwd() + '/csv_FY/'
weatherdir = os.getcwd() + '/csv_FY/weather/'

# label for plotting
ylabel_dict = {'combined':'Electricity + Gas [kBtu/sq.ft]',
               'elec':'Electricity Conditioning [kBtu/sq.ft]',
               'base':'Base Load [kBtu/sq.ft]',
               'gas':'Natural Gas Conditioning [kBtu/sq.ft]'}

title_dict = {'combined':'Electricity + Gas',
              'elec':'Electricity Conditioning',
              'base':'Base Load',
              'gas':'Gas Conditioning'}

def plot_lean_one(b, s, side, **kwargs):
    print 'creating {0} LEAN for building {1} ...'.format(side, b)
    sns.set_style("darkgrid")
    sns.set_palette("Set2")
    sns.set_context("talk", font_scale=1)
    gas_line_color = 'deeppink'
    gas_mk_color = 'crimson'
    elec_line_color = 'navy'
    elec_mk_color = 'deepskyblue'
    base_gas_color = 'orange'
    base_elec_color = 'yellow'
    bx = plt.axes()
    if side == 'elec' or side == 'gas':
        t = kwargs[side]['x_range']
        t_min = t[0]
        t_max = t[1]
        par = kwargs[side]['regression_par']
        k = par[0]
        intercept = par[1]
        if type(kwargs[side]['breakpoint']) == int:
            breakpoint = kwargs[side]['breakpoint']
        else:
            break_left = kwargs[side]['breakpoint'][0]
            break_right = kwargs[side]['breakpoint'][1]
            breakpoint = break_left
        base = k * breakpoint + intercept
        xd = np.linspace(t_min, t_max, 150)
        yd = kwargs[side]['fun'](xd, *par) - base
        if side == 'gas':
            plt.plot(xd, yd, gas_line_color)
            bx.fill_between(xd, 0, yd, facecolor=gas_line_color,
                            alpha=0.3)
            rug_x = kwargs[side]['x']
            rug_x = [x for x in rug_x if x < breakpoint]
            sns.rugplot(rug_x, ax=bx, color=gas_line_color)
        elif side == 'elec':
            plt.plot(xd, yd, elec_line_color)
            bx.fill_between(xd, 0, yd, facecolor=elec_line_color,
                            alpha=0.3)
            rug_x = kwargs[side]['x']
            try: 
                break_left
                rug_x = [x for x in rug_x if x > break_right or x <
                         break_left]
            except NameError:
                rug_x = [x for x in rug_x if x > breakpoint]
            sns.rugplot(rug_x, ax=bx, color=elec_line_color)
        plt.ylim((0, max(yd) * 1.1))
    else:
        t = kwargs['gas']['x_range']
        t_min = t[0]
        t_max = t[1]
        par_gas = kwargs['gas']['regression_par']
        par_elec = kwargs['elec']['regression_par']
        k_gas = par_gas[0]
        intercept_gas = par_gas[1]
        breakpoint_gas = kwargs['gas']['breakpoint']
        base_gas = k_gas * breakpoint_gas + intercept_gas
        k_elec = par_elec[0]
        intercept_elec = par_elec[1]
        if type(kwargs['elec']['breakpoint']) == int:
            breakpoint_elec = kwargs['elec']['breakpoint']
        else:
            break_elec_left = kwargs['elec']['breakpoint'][0]
            break_elec_right = kwargs['elec']['breakpoint'][1]
            breakpoint_elec = break_elec_left
        base_elec = k_elec * breakpoint_elec + intercept_elec
        xd = np.linspace(t_min, t_max, 150)
        yd_gas = (kwargs['gas']['fun'](xd, *par_gas)) + base_elec
        yd_elec = (kwargs['elec']['fun'](xd, *par_elec)) + base_gas
        if side == 'combined':
            plt.plot(xd, yd_gas, gas_line_color)
            plt.plot(xd, yd_elec, elec_line_color)
            bx.fill_between(xd, base_elec + base_gas, yd_elec,
                            facecolor=elec_line_color, alpha=0.3)
            bx.fill_between(xd, base_elec + base_gas, yd_gas,
                            facecolor=gas_line_color, alpha=0.3)
            plt.ylim((0, max(max(yd_elec), max(yd_gas)) * 1.1))
        elif side == 'base':
            plt.plot(xd, [base_elec] * len(xd), base_elec_color)
            plt.plot(xd, [base_elec + base_gas] * len(xd),
                     base_gas_color)
        bx.fill_between(xd, 0, base_elec,
                        facecolor=base_elec_color, alpha=0.5)
        bx.fill_between(xd, base_elec, base_elec + base_gas,
                        facecolor=base_gas_color, alpha=0.5)
        plt.ylim((0, max(max(yd_elec), max(yd_gas)) * 1.1))
        rug_x = kwargs['gas']['x']
        sns.rugplot(rug_x, ax=bx, color='gray')
    plt.title('Lean {0} plot, Building {1}, station {2}'.format(title_dict[side], b, s))
    plt.xlabel('Monthly Mean Temperature, Deg F')
    plt.ylabel(ylabel_dict[side])
    plt.tight_layout()
    P.savefig(os.getcwd() + '/plot_FY_weather/lean_piecewise/{0}_{1}_{2}.png'.format(b, s, side), dpi = 150, bbox_inches='tight')
    # plt.show()
    plt.close()

# adapted from Shilpi's code
def get_weather_data(s, minDate, maxDate):
    print 'start reading {0}'.format(s)
    starttime = time.time()
    url =  "https://128.2.109.159/piwebapi/dataservers/s0-MYhSMORGkyGTe9bdohw0AV0lOLTYyTlBVMkJWTDIw/points?namefilter=weatherunderground*{0}*temperature*Hourly".format(s)
    r = requests.get(url, auth=('Weather', 'Weather1!@'), verify=False)
    if len(r.json()['Items']) == 0:
        print 'No Data for station {0}'.format(s)
        return
    webId = r.json()['Items'][0]['WebId']
    recordUrl = "https://128.2.109.159/piwebapi/streams/"+webId+"/recorded?starttime='"+minDate+"'&endtime='"+maxDate+"'&maxcount=149000"
    rec = requests.get(recordUrl, auth=('Weather', 'Weather1!@'),
                       verify=False)
    json_list = (rec.json()['Items'])
    timestamps = [x['Timestamp'] for x in json_list]
    temp = [x['Value'] for x in json_list]
    df = pd.DataFrame({'Timestamp': timestamps, s: temp})
    # t1 = time.time()
    # df.to_csv(os.getcwd() + '/testoutput/{0}.csv'.format(s),
    #           index=False)
    print 'finish reading {0} in {1}s'.format(s, time.time() - starttime)
    return df

def get_mean_temp(s, minDate, maxDate): 
    df = get_weather_data(s, minDate, maxDate)
    df['localTime'] = pd.date_range(minDate, periods=len(df), freq='H')
    df.drop('Timestamp', axis=1, inplace=True)
    df.set_index(pd.DatetimeIndex(df['localTime']), inplace=True)
    df_re = df.resample('M', how = 'mean')
    df_re['year'] = df_re.index.map(lambda x: x.year)
    df_re['month'] = df_re.index.map(lambda x: x.month)
    print df_re.head()
    return df_re

# input: json_str of response
def parse_response(json_str):
    raise 'NotImplementedError'
    return
    
def get_lat_long(address):
    g = geocoder.google(address)
    if not (g.json['ok']):
        print 'Address not found'
        return None
    else:
        latlng = g.latlng
        # print '{0},{1},{2}'.format(address, latlng[0], latlng[1])
        return latlng

def get_station(latlng):
    if latlng == None:
        print 'No input to get_station'
        return None
    df_lookup = pd.read_csv(os.getcwd() + \
                            '/input/Weather Data Mapping to Use.csv')
    lat = latlng[0]
    lng = latlng[1]
    df_lookup['distance'] = df_lookup.apply(lambda r: vincenty((lat, lng), (r['Latitude'], r['Longitude']), miles=True), axis=1)
    min_distance = df_lookup['distance'].min()
    df_temp = df_lookup[df_lookup['distance'] == min_distance]
    icao = df_temp['StationID'].tolist()[0]
    distance = df_temp['distance'].tolist()[0]
    return (icao, distance)

# b: building ID, kwargs include the following keys: state_abbr(state
# abbreviation, address(street address), zipcode(zip code or postal
# code), city(name of city)
# return weather station ICAO
def geocode(b, **kwargs):
    print 'geocoding ...'
    d = {k:kwargs[k] for k in kwargs if k in ['address', 'city',
                                              'state', 'zipcode']}
    tokens = []
    for k in ['address', 'city', 'state', 'zipcode']:
        if k in kwargs:
            if kwargs[k] != None:
                tokens.append(kwargs[k])
    geocode_input = ','.join(tokens)
    # caching needed
    latlng = get_lat_long(geocode_input)
    icao, distance = get_station(latlng)
    print '{0}, latlng: {1}, station: {2}, distance: {3} mile'.format(geocode_input, latlng, icao, distance)
    return icao

def test_geocode():
    geocode('testbuilding', address='1620 V STREET NW', state='DC',
            city='Washington', zipcode='20009')
    
def test_get_weather_data():
    s = geocode('testbuilding', address='1620 V STREET NW',
                state='DC', city='Washington', zipcode='20009')
    minDate = '2007-10-01 00:00:00'
    maxDate = '2016-01-01 00:00:00'
    get_weather_data(s, minDate, maxDate)
    return
    
def get_area_elec(d_elec, n_par_elec):
    if d_elec == None:
        return (0, 0)
    t = d_elec['x_range']
    t_min = t[0]
    t_max = t[1]
    par_elec = d_elec['regression_par']
    k_elec = par_elec[0]
    intercept_elec = par_elec[1]
    if n_par_elec == 2:
        breakpoint_elec = d_elec['breakpoint']
        f_elec = lambda x: k_elec * x + intercept_elec
        base_elec = k_elec * breakpoint_elec + intercept_elec
        area_elec = (t_max - breakpoint_elec) * (f_elec(t_max) -
                                                 base_elec) / 2
    else:
        break_elec_left = d_elec['breakpoint'][0]
        break_elec_right = d_elec['breakpoint'][1]
        k_elec_right = par_elec[2]
        f_elec_left = lambda x: k_elec * x + intercept_elec
        base_elec = k_elec * break_elec_left + intercept_elec
        f_elec_right = lambda x: k_elec_right * (x - break_elec_right)\
            + f_elec_left(break_elec_left)
        area_elec = (break_elec_left - t_min) * (f_elec_left(t_min) - base_elec) / 2 + (t_max - break_elec_right) * (f_elec_right(t_max) - base_elec) / 2
    area_base_elec = (t_max - t_min) * base_elec
    print 'area of base electricity: {0}'.format(area_base_elec)
    print 'area of conditioning electricity: {0}'.format(area_elec)
    return (area_base_elec, area_elec)

def get_area_gas(d_gas):
    print 'computing area under gas curve ...'
    if d_gas == None:
        return (0, 0)
    t = d_gas['x_range']
    t_min = t[0]
    t_max = t[1]
    par_gas = d_gas['regression_par']
    k_gas = par_gas[0]
    intercept_gas = par_gas[1]
    breakpoint_gas = d_gas['breakpoint']
    f_gas = lambda x: k_gas * x + intercept_gas
    base_gas = f_gas(breakpoint_gas)
    area_base_gas = (t_max - t_min) * base_gas
    area_gas = (breakpoint_gas - t_min) * (f_gas(t_min) - base_gas) / 2
    print 'area of base gas: {0}'.format(area_base_gas)
    print 'area of conditioning gas: {0}'.format(area_gas)
    return (area_base_gas, area_gas)

def lean_temperature(b, s, n_par_elec, *args):
    if len(args) > 0:
        d_gas = piecewise_reg_one(b, s, 2, 'eui_gas', args[0])
        d_elec = piecewise_reg_one(b, s, n_par_elec, 'eui_elec', args[0])
    else:
        d_gas = piecewise_reg_one(b, s, 2, 'eui_gas')
        d_elec = piecewise_reg_one(b, s, n_par_elec, 'eui_elec')
    if d_gas == None or d_elec == None:
        print 'no gas or electric data'
        return
    (area_base_elec, area_elec) = get_area_elec(d_elec, n_par_elec)
    (area_base_gas, area_gas) = get_area_gas(d_gas)
    # plot_lean_one(b, s, "gas", gas=d_gas)
    # plot_lean_one(b, s, "elec", elec=d_elec)
    # plot_lean_one(b, s, "base", gas=d_gas, elec=d_elec)
    plot_lean_one(b, s, "combined", gas=d_gas, elec=d_elec)
    print d_gas.keys()
    d_gas.pop('fun', None)
    d_gas.pop('x', None)
    d_gas.pop('y', None)
    d_gas['regression_par'] = list(d_gas['regression_par'])
    # d_gas['x'] = list(d_gas['x'])
    # d_gas['y'] = list(d_gas['y'])
    d_gas['area_base_gas'] = area_base_gas
    d_gas['area_gas'] = area_gas
    d_elec.pop('fun', None)
    d_elec.pop('x', None)
    d_elec.pop('y', None)
    d_elec['regression_par'] = list(d_elec['regression_par'])
    # d_elec['x'] = list(d_elec['x'])
    # d_elec['y'] = list(d_elec['y'])
    d_elec['area_base_elec'] = area_base_elec
    d_elec['area_elec'] = area_elec
    d_elec['area_base_elec'] = area_base_elec
    d_elec['area_elec'] = area_elec
    # with open (os.getcwd() + \
    #            '/testoutput/json/{0}_gas.json'.format(b), 'w+') as wt:
    #     json.dump(d_gas, wt)
    # with open (os.getcwd() + \
    #            '/testoutput/json/{0}_elec.json'.format(b), 'w+') as wt:
    #     json.dump(d_elec, wt)
    for key in ['x_range', 'CV(RMSE)', 'regression_par', 'breakpoint']:
        print 'gas[\'{0}\']: {1}'.format(key, d_gas[key])
    for key in ['x_range', 'CV(RMSE)', 'regression_par', 'breakpoint']:
        print 'elec[\'{0}\']: {1}'.format(key, d_elec[key])
    return d_gas, d_elec

# CV(RMSE) cutoff: the baseline model shall have a maximum CV(RMSE) of
# 20% for energy use, These requirements are 25% and 35%,
# respectively, when 12 to 60 months of data will be used in computing
# savings.
def CVRMSE(x, y, p, f, n_par):
    y_hat = f(x, *p)
    n = len(x)
    return np.sqrt((np.subtract(y_hat, y) ** 2).sum() / (n - n_par))/np.array(y).mean()

# issue: from the document, it seems more years will need larger
# threshold for error (CV(RMSE)), no threshold is specified for > 60
# months

# if n_par == 2: return {'breakpoint': int, 'regression_par':
# array([k, intercept]), 'CV(RMSE)': float}
# if n_par == 3: return {'breakpoint': (int, int), 'regression_par':
# array([k1, intercept1, k2]), 'CV(RMSE)': float}
def piecewise_reg_one(b, s, n_par, theme, *args):
    if len(args) == 0:
        df = pd.read_csv(weatherdir + 'energy_temp/{0}_{1}.csv'.format(b, s))
    else:
        df = args[0]
    x = np.array(df[s])
    y = np.array(df[theme])
    if sum(y) == 0:
        print 'zero {0} energy'.format(theme)
        return None
    t_min = x.min()
    t_max = x.max()
    break_low = 40
    break_high = 81
    xd = np.linspace(t_min, t_max, 150)
    cvrmses = []
    ps = []
    if n_par == 2:
        breakpoints = range(break_low, break_high)
        for breakpoint in breakpoints:
            if theme == 'eui_gas':
                def piecewise_linear(x, k, intercept):
                    return np.piecewise(x, [x < breakpoint, x >= breakpoint], [lambda x:k * x + intercept, lambda x:k * breakpoint + intercept])
            elif theme == 'eui_elec':
                def piecewise_linear(x, k, intercept):
                    return np.piecewise(x, [x >= breakpoint, x < breakpoint], [lambda x:k * x + intercept, lambda x:k * breakpoint + intercept])
            p , e = optimize.curve_fit(piecewise_linear, x, y)
            cvrmse = CVRMSE(x, y, p, piecewise_linear, n_par)
            cvrmses.append(cvrmse)
            ps.append(p)
    elif n_par == 3:
        breakpoints = [(i, j) for i in range(break_low, break_high) for j in range(i + 1, break_high)]
        for (break_1, break_2) in breakpoints:
            def piecewise_linear(x, k1, b1, k2):
                x0 = break_1
                x1 = break_2
                y0 = k1 * x0 + b1
                y1 = y0
                return np.piecewise(x, [x < x0, x >= x1], [lambda x:k1 * x + b1, lambda x:k2 * (x - x1) + y1, lambda x:y0])
            p , e = optimize.curve_fit(piecewise_linear, x, y)
            cvrmse = CVRMSE(x, y, p, piecewise_linear, n_par)
            cvrmses.append(cvrmse)
            ps.append(p)
    result = sorted(zip(breakpoints, cvrmses, ps), key=lambda x: x[1])
    best = result[0]
    b_point_opt = best[0]
    p_opt = best[2]
    if n_par == 2:
        if theme == 'eui_gas':
            def piecewise_linear(x, k, intercept):
                return np.piecewise(x, [x < b_point_opt, x >= b_point_opt], [lambda x:k * x + intercept, lambda x:k * b_point_opt + intercept])
        else:
            def piecewise_linear(x, k, intercept):
                return np.piecewise(x, [x >= b_point_opt, x < b_point_opt], [lambda x:k * x + intercept, lambda x:k * b_point_opt + intercept])
    elif n_par == 3:
        def piecewise_linear(x, k1, b1, k2):
            x0 = b_point_opt[0]
            x1 = b_point_opt[1]
            y0 = k1 * x0 + b1
            y1 = y0
            return np.piecewise(x, [x < x0, x >= x1], [lambda x:k1 * x + b1, lambda x:k2 * (x - x1) + y1, lambda x:y0])
    # plt.plot(x, y, "o")
    # plt.plot(xd, piecewise_linear(xd, *p_opt))
    # plt.title('break point {0}F'.format(b_point_opt))
    # plt.show()
    # plt.close()
    if best[1] > 0.25:
        print 'exceeding max CV(RMSE)!'
    return {'breakpoint': b_point_opt, 'CV(RMSE)': best[1],
            'regression_par': p_opt, 'x_range': (t_min, t_max), 'fun':
            piecewise_linear, 'x': x, 'y': y}

def main():
    # test_get_weather_data()
    # 2 corresponds to the 3-parameter change point model
    lean_temperature('NM0050ZZ', 'KABQ', 2)
    # 3 corresponds to the 5-parameter change point model
    lean_temperature('MA0113ZZ', 'KCQX', 3)

# main()
