def get_scale_factor(file_name):
    if '2011' in file_name:
        year='2011'
    if '2012' in file_name:
        year='2012'
    # Luminosity of each year
    lumi12 = 11580. #FACTOR NEEDED TO COMPENSATE THAT FRACTION OF THE EVENTS ARE CUT TO TRAIN THE MODELS
    lumi11 = 2330.

    # MC cross section of each process
    xsecZZ412 = 0.107
    xsecZZ2mu2e12 = 0.249
    xsecZZ411 = 0.093
    xsecZZ2mu2e11 = 0.208

    xsecTTBar12 = 200.
    xsecTTBar11 = 19.504

    xsecDY5012 = 2955.
    xsecDY1012 = 10.742
    xsecDY5011 = 2475.
    xsecDY1011 = 9507.

    scalexsecHZZ12 = 0.0065
    scalexsecHZZ11 = 0.0057

    # Number of MC Events generated for each process
    nevtZZ4mu12 = 1499064
    nevtZZ4e12 = 1499093
    nevtZZ2mu2e12 = 1497445
    nevtHZZ12 = 299973
    nevtTTBar12 = 6423106
    nevtDY5012 = 29426492
    nevtDY1012 = 6462290

    nevtZZ4mu11 = 1447136
    nevtZZ4e11 = 1493308
    nevtZZ2mu2e11 = 1479879
    nevtHZZ11 = 299683
    nevtTTBar11 = 9771205
    nevtDY5011 = 36408225
    nevtDY1011 = 39909640

    # Define scaling factors
    scales_higgs = [lumi11 * scalexsecHZZ11 / nevtHZZ11, lumi12 * scalexsecHZZ12 / nevtHZZ12]
    scales_zz = [lumi11 * xsecZZ411 / nevtZZ4mu11, lumi11 * xsecZZ2mu2e11 / nevtZZ2mu2e11,
                 lumi11 * xsecZZ411 / nevtZZ4e11, lumi12 * xsecZZ412 / nevtZZ4mu12,
                 lumi12 * xsecZZ2mu2e12 / nevtZZ2mu2e12, lumi12 * xsecZZ412 / nevtZZ4e12]
    scales_dy = [lumi11 * xsecDY1011 / nevtDY1011, lumi11 * xsecDY5011 / nevtDY5011,
                 lumi12 * xsecDY1012 / nevtDY1012, lumi12 * xsecDY5012 / nevtDY5012]
    scales_tt = [lumi11 * xsecTTBar11 / nevtTTBar11, lumi12 * xsecTTBar12 / nevtTTBar12]

    # Check which scale factor applies based on the filename and year
    if "higgs" in file_name:
        return scales_higgs[0] if year == '2011' else scales_higgs[1]
    elif "zz" in file_name:
        if year == '2011':
            if "4mu" in file_name:
                return scales_zz[0]
            elif "2mu2e" in file_name:
                return scales_zz[1]
            elif "4e" in file_name:
                return scales_zz[2]
        else:
            if "4mu" in file_name:
                return scales_zz[3]
            elif "2mu2e" in file_name:
                return scales_zz[4]
            elif "4e" in file_name:
                return scales_zz[5]
    elif "dy50" in file_name:
        return scales_dy[1] if year == '2011' else scales_dy[3]
    elif "dy10" in file_name:
        return scales_dy[0] if year == '2011' else scales_dy[2]
    elif "ttbar" in file_name:
        return scales_tt[0] if year == '2011' else scales_tt[1]
    else:
        return 1.0  # Default scaling factor (no scaling)