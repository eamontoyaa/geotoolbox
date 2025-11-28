import os
import subprocess  # import run, call
import shutil
import textwrap
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def download_file_from_github(file, url, output_path, make_executable=False):
    os.makedirs(output_path, exist_ok=True)
    destination = os.path.join(output_path, file)

    # ✅ Check if file already exists
    if os.path.exists(destination):
        print(f"🟡 File already exists at: {destination} — skipping download.")
        return

    url = f"{url.rstrip('/')}/{file}"

    try:
        print(f"⬇️ Downloading from: {url}")
        subprocess.run(["wget", "-q", "-O", destination, url], check=True)

        if make_executable:
            os.chmod(destination, 0o755)
            print(f"🔧 File marked as executable.")

        print(f"✅ File downloaded to: {destination}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")


def tpx_in_maker(tpx_inputs, proj_folder, print_output=True):
    """
    Generates the input file for the TopoIndex executable.

    Args:
        tpx_inputs (dict): Dictionary containing required input parameters.
        proj_folder (str): Project folder path where the input file will be stored.
    """
    try:
        # Define file paths
        input_file_path = "tpx_in.txt"  # os.path.join(proj_folder, "tpx_in.txt")
        inputs_dir = os.path.join(proj_folder, "inputs")

        # Ensure the inputs directory exists
        os.makedirs(inputs_dir, exist_ok=True)

        # Template for the input file
        tpx_in_template = textwrap.dedent(
            f"""\
            TopoIndex 1.0.15; Name of project (up to 255 characters)
            TopoIndex analysis - Project: {tpx_inputs.get("proj_id", "UNKNOWN_PROJECT")}
            Flow-direction numbering scheme (ESRI=1, TopoIndex=2)
            {tpx_inputs.get("flow_dir_scheme", 1)}
            Exponent, Number of iterations
            {tpx_inputs.get("exponent_weight_fact", 1.0)}, {tpx_inputs.get("iterations", 100)}
            Name of elevation grid file
            {os.path.join(inputs_dir, "dem.asc")}
            Name of direction grid
            {os.path.join(inputs_dir, "directions.asc")}
            Save listing of D8 downslope neighbor cells (TIdsneiList_XYZ)?  Enter T (.true.) or F (.false.)
            T
            Save grid of D8 downslope neighbor cells (***TIdscelGrid_XYZ)? Enter T (.true.) or F (.false.)
            T
            Save cell index number grid (TIcelindxGrid_XYZ)?  Enter T (.true.) or F (.false.)
            T
            Save list of cell number and corresponding index number (***TIcelindxList_XYZ)? Enter T (.true.) or F (.false.)
            T
            Save flow-direction grid remapped from ESRI to TopoIndex (TIflodirGrid_XYZ)? Enter T (.true.) or F (.false.)
            T
            Save grid of points on ridge crests (TIdsneiList_XYZ)? Enter T (.true.) or F (.false.); Sparse (T) or dense (F)?
            F,T
            ID code for output files? (8 characters or less)
            {tpx_inputs.get("proj_id", "UNKNOWN")}
        """
        )

        # Write to file
        with open(input_file_path, "w") as file:
            file.write(tpx_in_template)
        if print_output:
            print(
                f"Input file successfully created at: {os.path.join(os.getcwd(), input_file_path)}\n"
            )
    except Exception as e:
        print(f"Error generating input file: {e}")


def run_topoidx(tpx_inputs, proj_folder, print_output=True):
    """
    Run the TopoIndex executable after generating input files, setting permissions,
    and managing output logs.

    Args:
        tpx_inputs (dict): Dictionary containing input parameters, including 'proj_id'.
        proj_folder (str): Project folder name.
    """
    try:
        # Define paths
        workdir = os.getcwd()
        exec_dir = os.path.join(workdir, proj_folder)
        outputs_dir = os.path.join(exec_dir, "outputs")

        # Ensure directories exist
        os.makedirs(outputs_dir, exist_ok=True)

        # Generate initialization file
        tpx_in_maker(tpx_inputs, proj_folder, print_output)

        # Define executable path
        executable_path = os.path.join(exec_dir, "Exe_TopoIndex")
        # Ensure the executable has the correct permissions
        subprocess.run(["chmod", "+x", executable_path], check=True)

        # Run the Fortran executable
        if print_output:
            subprocess.run([executable_path], check=True)
        else:
            subprocess.run(
                [executable_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

        # Move and rename output files
        log_src = os.path.join(workdir, "TopoIndexLog.txt")
        log_dest = os.path.join(
            outputs_dir, f"TopoIndexLog_{tpx_inputs['proj_id']}.txt"
        )
        input_src = os.path.join(workdir, "tpx_in.txt")
        input_dest = os.path.join(exec_dir, f"tpx_in_{tpx_inputs['proj_id']}.txt")

        shutil.move(log_src, log_dest)
        shutil.move(input_src, input_dest)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Execution of {e.cmd} failed with exit code {e.returncode}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return


def tr_in_maker(trg_inputs, proj_folder, print_output=True):
    """
    Generates the input file for the TRIGRS executable.

    Args:
        trg_inputs (dict): Dictionary containing required input parameters.
        proj_folder (str): Project folder path where the input file will be stored.
    """
    try:
        # Define file paths
        input_file_path = "tr_in.txt"  # os.path.join(proj_folder, "tr_in.txt")
        inputs_dir = os.path.join(proj_folder, "inputs")

        # Ensure the inputs directory exists
        os.makedirs(inputs_dir, exist_ok=True)

        # Create text block for geotechnical parameters of each zone
        geoparams = ""
        for zone, z in trg_inputs.get("geoparams", {}).items():
            txt_zone = "\n".join(
                [
                    f"zone, {zone}",
                    "cohesion, phi, uws, diffus, K-sat, Theta-sat, Theta-res, Alpha",
                    f'{z.get("c", 0)}, {z.get("φ", 0)}, {z.get("γ_sat", 0)}, {z.get("D_sat", 0)}, '
                    f'{z.get("K_sat", 0)}, {z.get("θ_sat", 0)}, {z.get("θ_res", 0)}, {z.get("α", 0)}',
                ]
            )
            geoparams += f"{txt_zone}\n"

        # Update rifil entry with placeholders for each file
        rifil_placeholders = (
            "\n".join(
                [
                    os.path.join(inputs_dir, f"ri{n}.asc")
                    for n in range(1, len(trg_inputs.get("capt", [])))
                ]
            )
            + "\n"
        )

        # Template for the input file
        trg_in_template = (
            textwrap.dedent(
                f"""\
            Project: {trg_inputs.get('proj_id', 'UNKNOWN_PROJECT')}
            TRIGRS, version 2.1.00c, (meter-kilogram-second-degrees units)
            tx, nmax, mmax, zones
            {trg_inputs.get('tx', 0)}, {trg_inputs.get('nmax', 0)}, {trg_inputs.get('mmax', 0)}, {trg_inputs.get('zones', 0)}
            nzs, zmin, uww, nper t
            {trg_inputs.get('nzs', 0)}, 0.001, 9.8e3, {trg_inputs.get('nper', 0)}, {trg_inputs.get('t', 0)}
            zmax, depth, rizero, Min_Slope_Angle (degrees), Max_Slope_Angle (degrees)
            {trg_inputs.get('zmax', 0)}, {trg_inputs.get('depth', 0)}, {trg_inputs.get('rizero', 0)}, 0., 90.0
        """
            )
            + geoparams
            + textwrap.dedent(
                f"""\
            cri(1), cri(2), ..., cri(nper)
            {', '.join(map(str, trg_inputs.get('cri', [])))}
            capt(1), capt(2), ..., capt(n), capt(n+1)
            {', '.join(map(str, trg_inputs.get('capt', [])))}
            File name of slope angle grid (slofil)
            {os.path.join(inputs_dir, 'slope.asc')}
            File name of digital elevation grid (elevfil)
            {os.path.join(inputs_dir, 'dem.asc')}
            File name of property zone grid (zonfil)
            {os.path.join(inputs_dir, 'zones.asc')}
            File name of depth grid (zfil)
            {os.path.join(inputs_dir, 'zmax.asc')}
            File name of initial depth of water table grid (depfil)
            {os.path.join(inputs_dir, 'depthwt.asc')}
            File name of initial infiltration rate grid (rizerofil)
            {os.path.join(inputs_dir, 'rizero.asc')}
            List of file names of rainfall intensity for each period (rifil())
        """
            )
            + rifil_placeholders
            + textwrap.dedent(
                f"""\
            File name of grid of D8 runoff receptor cell numbers (nxtfil)
            {os.path.join(inputs_dir, f"TIdscelGrid_{trg_inputs.get('proj_id', 'UNKNOWN')}.txt")}
            File name of list of defining runoff computation order (ndxfil)
            {os.path.join(inputs_dir, f"TIcelindxList_{trg_inputs.get('proj_id', 'UNKNOWN')}.txt")}
            File name of list of all runoff receptor cells (dscfil)
            {os.path.join(inputs_dir, f"TIdscelList_{trg_inputs.get('proj_id', 'UNKNOWN')}.txt")}
            File name of list of runoff weighting factors (wffil)
            {os.path.join(inputs_dir, f"TIwfactorList_{trg_inputs.get('proj_id', 'UNKNOWN')}.txt")}
            Folder where output grid files will be stored (folder)
            {os.path.join(proj_folder, 'outputs/')}
            Identification code to be added to names of output files (suffix)
            {trg_inputs.get('proj_id', 'UNKNOWN')}
            Save grid files of runoff? Enter T (.true.) or F (.false.)
            F
            Save grid of minimum factor of safety? Enter T (.true.) or F (.false.)
            T
            Save grid of depth of minimum factor of safety? Enter T (.true.) or F (.false.)
            T
            Save grid of pressure head at depth of minimum factor of safety? Enter T (.true.) or F (.false.)
            T
            Save grid of computed water table depth or elevation? Enter T (.true.) or F (.false.) followed by 'depth' or 'eleva'
            T, depth
            Save grid files of actual infiltration rate? Enter T (.true.) or F (.false.)
            F
            Save grid files of unsaturated zone basal flux? Enter T (.true.) or F (.false.)
            F
            Save listing of pressure head and factor of safety ("flag")? Enter flag value followed by down-sampling interval.
            {trg_inputs.get('ψ_flag', 0)}, {trg_inputs.get('nzs', 0)}
            Number of times to save output grids and (or) ijz/xmdv files
            {trg_inputs.get('n_outputs', 0)}
            Times of output grids and (or) ijz/xmdv files
            {', '.join(map(str, trg_inputs.get('t_n_outputs', [])))}
            Skip other timesteps? Enter T (.true.) or F (.false.)
            F
            Use analytic solution for fillable porosity? Enter T (.true.) or F (.false.)
            T
            Estimate positive pressure head in rising water table zone? Enter T (.true.) or F (.false.)
            T
            Use psi0=-1/alpha? Enter T (.true.) or F (.false.)
            T
            Log mass balance results? Enter T (.true.) or F (.false.)
            T
            Flow direction (Enter "gener", "slope", or "hydro")
            {trg_inputs.get('flowdir', 'slope')}
            Add steady background flux to transient infiltration rate?
            T
            Specify file extension for output grids. Enter T (.true.) for ".asc" or F for ".txt"
            T
            Ignore negative pressure head in computing factor of safety? Enter T (.true.) or F (.false.)
            T
            Ignore height of capillary fringe in computing pressure head for unsaturated infiltration? Enter T (.true.) or F (.false.)
            T
            Parameters for deep pore-pressure estimate: Depth below ground surface, pressure option ('zero', 'flow', 'hydr', or 'relh')
            -50.0, flow
        """
            )
        )

        # Write to file
        with open(input_file_path, "w") as file:
            file.write(trg_in_template)
        if print_output:
            print(
                f"Input file successfully created at: {os.path.join(os.getcwd(), input_file_path)}"
            )

    except Exception as e:
        print(f"Error generating input file: {e}\n")


def run_trigrs(trg_inputs, proj_folder, print_output=True):
    """
    Runs the TRIGRS executable after generating input files, setting permissions,
    and organizing output files.

    Args:
        trg_inputs (dict): Dictionary containing input parameters.
        proj_folder (str): Name of the project folder.
    """
    try:
        # Define paths
        workdir = os.getcwd()
        exec_dir = os.path.join(workdir, proj_folder)
        outputs_dir = os.path.join(exec_dir, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        # Correcting sign for `Alpha` parameter for saturated/unsaturated models
        for z, zone in trg_inputs.get("geoparams", {}).items():
            if "S" in trg_inputs.get("model", ""):
                zone["α"] = -abs(zone["α"])
            else:
                zone["α"] = abs(zone["α"])

        # Correcting sign for `mmax` parameter for finite/infinite models
        trg_inputs["mmax"] = (
            -abs(trg_inputs["mmax"])
            if "I" in trg_inputs.get("model", "")
            else abs(trg_inputs["mmax"])
        )

        # Generate initialization file
        tr_in_maker(trg_inputs, proj_folder, print_output)

        # Select executable based on parallel processing setting
        if trg_inputs.get("parallel", False):
            executable_path = os.path.join(exec_dir, "Exe_TRIGRS_par")
            # Ensure the executable has the correct permissions
            subprocess.run(["chmod", "+x", executable_path], check=True)
            # Run TRIGRS executable
            if print_output:
                subprocess.run(
                    ["mpirun", "-np", str(trg_inputs["NP"]), executable_path]
                )
            else:
                subprocess.run(
                    ["mpirun", "-np", str(trg_inputs["NP"]), executable_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        else:
            executable_path = os.path.join(exec_dir, "Exe_TRIGRS_ser")
            # Ensure the executable has the correct permissions
            subprocess.run(["chmod", "+x", executable_path], check=True)
            # Run TRIGRS executable
            if print_output:
                subprocess.run([executable_path], check=True)
            else:
                subprocess.run(
                    [executable_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        # Move log and input files
        proj_id, model = trg_inputs.get("proj_id", "UNKNOWN"), trg_inputs.get(
            "model", "DEFAULT"
        )
        log_src = os.path.join(workdir, "TrigrsLog.txt")
        log_dest = os.path.join(outputs_dir, f"TrigrsLog_{proj_id}_{model}.txt")
        input_src = os.path.join(workdir, "tr_in.txt")
        input_dest = os.path.join(exec_dir, f"tr_in_{proj_id}_{model}.txt")

        os.rename(log_src, log_dest)
        os.rename(input_src, input_dest)

        # Organizing output folders
        output_groups = [
            "TRfs_min",
            "TRp_at_fs_min",
            "TRwater_depth",
            "TRz_at_fs_min",
            "TR_ijz_p_th",
            # "TRrunoffPer",
        ]
        for grp in output_groups:
            os.makedirs(os.path.join(outputs_dir, grp), exist_ok=True)

        # Rename output files based on their group
        idx = np.arange(1, trg_inputs.get("n_outputs", 1) + 1)
        for grp, i in itertools.product(output_groups, idx):
            src_file, dest_file = "", ""

            if grp == "TRrunoffPer":
                src_file = os.path.join(outputs_dir, f"{grp}{i}{proj_id}.asc")
                dest_file = os.path.join(
                    outputs_dir, grp, f"{grp}_{proj_id}_{model}_{i}.asc"
                )
            elif grp == "TR_ijz_p_th":
                src_file = os.path.join(outputs_dir, f"{grp}_{proj_id}_{i}.txt")
                dest_file = os.path.join(
                    outputs_dir, grp, f"{grp}_{proj_id}_{model}_{i}.txt"
                )
            else:
                src_file = os.path.join(outputs_dir, f"{grp}_{proj_id}_{i}.asc")
                dest_file = os.path.join(
                    outputs_dir, grp, f"{grp}_{proj_id}_{model}_{i}.asc"
                )

            if os.path.exists(src_file):
                os.rename(src_file, dest_file)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Execution of {e.cmd} failed with exit code {e.returncode}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def get_param(trg_inputs, zone, param):
    return trg_inputs["geoparams"][zone][param]


def get_df(file, trg_inputs, ZONES, SLOPE):
    """
    Processes a TRIGRS output file and computes geotechnical parameters,
    slopes, and the factor of safety (Fs) for each grid cell.

    Args:
        file (str): Path to the TRIGRS output file.
        trg_inputs (dict): Dictionary containing geotechnical parameters.
        ZONES (numpy.ndarray): 2D array of zone indices.
        SLOPE (numpy.ndarray): 2D array of slope angles in degrees.

    Returns:
        pd.DataFrame: Processed data containing geotechnical properties and Fs values.
    """
    γ_w = 9.8e3  # [N/m³] Unit weight of water

    # Read the file into a DataFrame
    df = pd.read_csv(
        file, names=["i_tr", "j_tr", "elev", "ψ", "θ"], skiprows=7, sep=r"\s+"
    )

    # Convert TRIGRS indices to match NumPy notation
    df["i"] = df["j_tr"].max() - df["j_tr"]
    df["j"] = df["i_tr"] - 1
    df.drop(columns=["i_tr", "j_tr"], inplace=True)

    # Compute depth (z) from elevation
    elev_max = df.groupby(["i", "j"])["elev"].transform("max")
    df["z"] = elev_max - df["elev"] + 0.001
    df = df[["i", "j", "elev", "z", "ψ", "θ"]]

    # Assign zones and geotechnical parameters
    df["zone"] = ZONES[df["i"], df["j"]]

    # Function to retrieve geotechnical parameters
    def get_zone_params(row):
        zone_id = row["zone"]
        params = trg_inputs.get("geoparams", {}).get(zone_id, {})
        return pd.Series(
            {
                param: params.get(param, np.nan)
                for param in [
                    "c",
                    "φ",
                    "γ_sat",
                    "D_sat",
                    "K_sat",
                    "θ_sat",
                    "θ_res",
                    "α",
                ]
            }
        )

    # Apply geotechnical parameters efficiently
    df = df.join(df.apply(get_zone_params, axis=1))

    # Correct numerical errors for ψ=0 at z > 0.001
    mask = (df["ψ"] == 0) & (df["z"] > 0.001) & (df["θ"] == df["θ_sat"])
    df.drop(df[mask].index, inplace=True)  # Removes incorrect watertable rows

    # Assign slope values
    df["δ"] = SLOPE[df["i"], df["j"]]
    δrad = np.deg2rad(df["δ"])  # Convert slope to radians

    # Compute effective stress parameter, χ
    df["χ"] = (df["θ"] - df["θ_res"]) / (df["θ_sat"] - df["θ_res"])

    # Compute specific gravity (Gs)
    df["Gs"] = (df["γ_sat"] / γ_w - df["θ_sat"]) / (1 - df["θ_sat"])

    # Compute unit weight in unsaturated and saturated zones
    df["γ_nat"] = (df["Gs"] * (1 - df["θ_sat"]) + df["θ"]) * γ_w

    # Compute average unit weight
    df["γ_cumul"] = df.groupby(["i", "j"])["γ_nat"].transform("cumsum")
    df["idx"] = df.groupby(["i", "j"])["γ_nat"].cumcount() + 1
    df["γ_avg"] = df["γ_cumul"] / df["idx"]
    df.drop(columns=["γ_cumul", "idx"], inplace=True)

    # Compute Factor of Safety (Fs)
    fs1 = np.tan(np.deg2rad(df["φ"])) / np.tan(δrad)
    fs2_num = df["c"] - df["ψ"] * γ_w * df["χ"] * np.tan(np.deg2rad(df["φ"]))
    fs2_den = df["γ_avg"] * df["z"] * np.sin(δrad) * np.cos(δrad)
    df["fs"] = fs1 + fs2_num / fs2_den
    df["fs"] = np.clip(df["fs"], 0, 3)  # Limit Fs values to a maximum of 3

    return df


def read_time(file):
    """
    Reads the time value from the 5th line of a given file.

    Args:
        file (str): Path to the file.

    Returns:
        float: Extracted time value.

    Raises:
        ValueError: If the file does not contain enough lines or the time value cannot be converted.
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(file, "r") as f:
            lines = f.readlines()

        if len(lines) < 5:
            raise ValueError(f"File {file} does not contain enough lines.")

        time_value = lines[4].strip().split()[-1]  # Get last element from 5th line
        return float(time_value)

    except FileNotFoundError:
        print(f"Error: The file '{file}' was not found.")
    except ValueError as e:
        print(f"Error processing file '{file}': {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None  # Return None if an error occurs


def create_asc_files_1cell(exec_dir):
    for file in ["dem", "zmax", "depthwt", "slope", "directions", "zones"]:
        with open(f"{exec_dir}/inputs/{file}.asc", "w") as asc:
            asc.write("ncols         1\n")
            asc.write("nrows         1\n")
            asc.write("xllcorner     0\n")
            asc.write("yllcorner     0\n")
            asc.write("cellsize      1\n")
            asc.write("NODATA_value  -9999\n")
            asc.write("0\n")


def set_val_asc_file_1cell(exec_dir, file, new_val):
    with open(f"{exec_dir}/inputs/{file}.asc", "r") as asc:
        lines = asc.readlines()
    with open(f"{exec_dir}/inputs/{file}.asc", "w") as asc:
        lines[6] = str(new_val) + "\n"
        asc.writelines(lines)
    return
