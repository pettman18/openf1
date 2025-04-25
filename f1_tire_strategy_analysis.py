#!/usr/bin/env python3
"""
F1 Pace and Tire Strategy Analysis Script (2024 Season)

Connects to MongoDB, fetches lap and stint data for 2024 F1 sessions.
Analyzes:
- Most common starting tire compounds.
- Typical tire life percentiles (based on stint length).
- Most common tire usage sequences.
- Correlation between FP2 pace (avg lap time) and Race pace per compound.
Outputs results to console, JSON, and CSV files.
"""

import os
import json
import pandas as pd
from pymongo import MongoClient
from scipy.stats import spearmanr
import traceback
import numpy as np # For handling potential NaN in JSON

# MongoDB connection settings
MONGO_CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
DB_NAME = "openf1-livetiming"
OUTPUT_DIR = "output/2024_analysis" # Unified output directory

# --- Configuration ---
# Percentiles for lap time filtering (removes outliers like SC/pit laps)
# Keeps laps between lower and upper percentile within each group
LAP_PACE_FILTER_LOWER_PERCENTILE = 0.10 # e.g., remove slowest 10%
LAP_PACE_FILTER_UPPER_PERCENTILE = 0.95 # e.g., remove fastest 5% (often errors or anomalies)
MIN_LAPS_FOR_CORRELATION = 3 # Minimum number of data points (drivers) needed per group for correlation


def connect_to_mongodb():
    """Connect to MongoDB and return the database object."""
    print(f"Connecting to MongoDB ({MONGO_CONNECTION_STRING})...")
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    print("Connected successfully!")
    return db

def print_all_2024_session_names(db):
    """Prints distinct session names for the year 2024."""
    try:
        session_names = db.sessions.distinct("session_name", {"year": 2024})
        print("\nDistinct Session names found for 2024:", session_names)
    except Exception as e:
        print(f"Could not fetch session names: {e}")

# <<< NEW FUNCTION to fetch LAP data >>>
def get_lap_data(db, session_type=None, session_name=None):
    """
    Fetch lap data for specified 2024 sessions, joining to get compound info.

    Args:
        db: MongoDB database object
        session_type (str, optional): Filter by session type (e.g., "Race", "Practice").
        session_name (str, optional): Filter by specific session name (e.g., "Practice 2").

    Returns:
        List of documents with lap data including compound.
    """
    session_match_criteria = {"session_info.year": 2024}
    if session_type:
        session_match_criteria["session_info.session_type"] = session_type
    if session_name:
        session_match_criteria["session_info.session_name"] = session_name

    print(f"Fetching LAP data for 2024 sessions matching: {session_match_criteria}...")

    pipeline = [
        # 1. Start with Laps, basic filter for existing duration
        {"$match": {"lap_duration": {"$ne": None, "$exists": True}}},

        # 2. Join with Sessions
        {
            "$lookup": {
                "from": "sessions",
                "localField": "session_key",
                "foreignField": "session_key",
                "as": "session_info"
            }
        },
        {"$unwind": "$session_info"},
        {"$match": session_match_criteria}, # Apply session filters here

        # 3. Join with Meetings
        {
            "$lookup": {
                "from": "meetings",
                "localField": "session_info.meeting_key",
                "foreignField": "meeting_key",
                "as": "meeting_info"
            }
        },
        {"$unwind": "$meeting_info"},

        # 4. Join with Stints to get Compound for this lap
        {
            "$lookup": {
                "from": "stints",
                "let": {
                    "lap_session_key": "$session_key",
                    "lap_driver": "$driver_number",
                    "lap_num": "$lap_number"
                },
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$session_key", "$$lap_session_key"]},
                                    {"$eq": ["$driver_number", "$$lap_driver"]},
                                    {"$lte": ["$lap_start", "$$lap_num"]},
                                    # If lap_end is null (last stint), assume lap belongs if lap_start matches
                                    {"$or": [
                                        {"$gte": ["$lap_end", "$$lap_num"]},
                                        {"$eq": ["$lap_end", None]} # Include laps from ongoing stints
                                    ]}
                                ]
                            }
                        }
                    },
                    # Sort by lap_start descending in case lap number overlaps
                    # between stints (unlikely but safety measure)
                    {"$sort": {"lap_start": -1}},
                    {"$limit": 1}, # Take the most recent matching stint
                    {"$project": {"_id": 0, "compound": 1}}
                ],
                "as": "stint_info"
            }
        },
        # Use compound from the first (and only) stint match, or None if no match
        {"$addFields": {"stint_compound": {"$ifNull": [{"$first": "$stint_info.compound"}, None]}}},

        # 5. Project final fields
        {
            "$project": {
                "_id": 0,
                "race_name": "$meeting_info.meeting_name",
                "race_date": "$session_info.date_start",
                "circuit": "$meeting_info.circuit_short_name",
                "location": "$meeting_info.location",
                "session_key": "$session_key",
                "driver_number": "$driver_number",
                "lap_number": "$lap_number",
                "lap_duration": "$lap_duration",
                "compound": "$stint_compound",
                # Include pit flags if they exist in your 'laps' collection
                "is_pit_in_lap": {"$ifNull": ["$is_pit_in_lap", False]},
                "is_pit_out_lap": {"$ifNull": ["$is_pit_out_lap", False]}
            }
        }
    ]

    try:
        results = list(db.laps.aggregate(pipeline))
        print(f"Found {len(results)} lap records.")
        return results
    except Exception as e:
        print(f"Error fetching lap data: {e}")
        print("Please ensure 'laps', 'sessions', 'meetings', 'stints' collections exist and have expected fields.")
        return []


# --- Stint-based Analysis Functions (Keep original logic) ---

def get_stint_data(db, session_type="Race", session_name=None):
    """ Fetches STINT data (Unchanged from previous version). """
    match_criteria = {"session.year": 2024}
    if session_type:
        match_criteria["session.session_type"] = session_type
    if session_name:
        match_criteria["session.session_name"] = session_name
    print(f"Fetching STINT data for 2024 sessions matching: {match_criteria}...")
    pipeline = [
        {"$lookup": {"from": "sessions", "localField": "session_key", "foreignField": "session_key", "as": "session"}},
        {"$unwind": "$session"}, {"$match": match_criteria},
        {"$lookup": {"from": "meetings", "localField": "meeting_key", "foreignField": "meeting_key", "as": "meeting"}},
        {"$unwind": "$meeting"},
        {"$project": {
            "_id": 0, "race_name": "$meeting.meeting_name", "race_date": "$session.date_start",
            "circuit": "$meeting.circuit_short_name", "location": "$meeting.location",
            "driver_number": "$driver_number", "compound": "$compound", "lap_start": "$lap_start",
            "lap_end": "$lap_end", "stint_number": "$stint_number",
            "laps_on_tire": {"$cond": {
                "if": {"$and": [{"$ne": ["$lap_start", None]}, {"$ne": ["$lap_end", None]}]},
                "then": {"$add": [{"$subtract": ["$lap_end", "$lap_start"]}, 1]}, "else": None}}
        }}
    ]
    results = list(db.stints.aggregate(pipeline))
    print(f"Found {len(results)} stint records.")
    return results

def analyze_tire_strategies(data):
    """ Analyzes STINT data for starting tire and tire life percentiles (Unchanged). """
    print("Analyzing tire strategies (based on stints)...")
    df = pd.DataFrame(data)
    if df.empty: return {"strategy": {}, "percentiles": {}}
    if 'race_date' in df.columns:
        df['race_date'] = pd.to_datetime(df['race_date'])
        df = df.sort_values(by=['race_date', 'driver_number', 'stint_number'])
    else: df = df.sort_values(by=['driver_number', 'stint_number'])

    # Starting tire
    if 'stint_number' in df.columns:
        starting_stints_df = df[df['stint_number'] == 1].copy()
    else:
        df['lap_start'] = pd.to_numeric(df['lap_start'], errors='coerce')
        df_valid_starts = df.dropna(subset=['lap_start'])
        starting_stints_df = df_valid_starts.loc[df_valid_starts.groupby(['race_name', 'driver_number'])['lap_start'].idxmin()] if not df_valid_starts.empty else pd.DataFrame()

    most_common_strategy = {}
    if not starting_stints_df.empty:
        mode_results = starting_stints_df.groupby('race_name')['compound'].agg(lambda x: x.mode().tolist())
        most_common_strategy = mode_results.apply(lambda x: x[0] if x else 'Unknown').to_dict()

    # Percentiles
    df['laps_on_tire'] = pd.to_numeric(df['laps_on_tire'], errors='coerce')
    df_valid_laps = df.dropna(subset=['laps_on_tire'])
    percentile_analysis = {}
    if not df_valid_laps.empty:
        percentiles_data = df_valid_laps.groupby(['race_name', 'compound'])['laps_on_tire'].quantile([0.4, 0.8]).unstack().reset_index()
        percentiles_data = percentiles_data.rename(columns={0.4: '40th_percentile_laps', 0.8: '80th_percentile_laps'})
        for race_name, race_data in percentiles_data.groupby('race_name'):
            percentile_analysis[race_name] = race_data.fillna(np.nan)[['compound', '40th_percentile_laps', '80th_percentile_laps']].to_dict('records')

    return {"strategy": most_common_strategy, "percentiles": percentile_analysis}

def analyze_tire_sequences(data):
    """ Analyzes STINT data for common tire sequences (Unchanged). """
    print("Analyzing tire sequences (based on stints)...")
    df = pd.DataFrame(data)
    if df.empty: return {}
    if 'race_date' in df.columns:
        df['race_date'] = pd.to_datetime(df['race_date'])
        df = df.sort_values(by=['race_date', 'driver_number', 'stint_number'])
    elif 'stint_number' in df.columns: df = df.sort_values(by=['driver_number', 'stint_number'])
    else:
        df['lap_start'] = pd.to_numeric(df['lap_start'], errors='coerce')
        df = df.dropna(subset=['lap_start']).sort_values(by=['driver_number', 'lap_start'])

    df['compound'] = df['compound'].astype(str)
    sequences = df.groupby(['race_name', 'driver_number'])['compound'].apply(list).reset_index()
    sequences['compound_sequence'] = sequences['compound'].apply(lambda seq: [v for i, v in enumerate(seq) if i == 0 or v != seq[i - 1]])

    most_common_sequences = {}
    if not sequences.empty:
        sequences['compound_sequence_tuple'] = sequences['compound_sequence'].apply(tuple)
        mode_results = sequences.groupby('race_name')['compound_sequence_tuple'].agg(lambda x: x.mode().tolist())
        most_common_sequences = mode_results.apply(lambda x: list(x[0]) if x else []).to_dict()

    return most_common_sequences


# --- Pace-based Analysis Functions (MODIFIED logic) ---

def filter_lap_data_for_pace_analysis(df, group_keys):
    """Filters lap DataFrame to remove outliers and invalid data for pace analysis."""
    print(f"  Initial lap count: {len(df)}")
    df['lap_duration'] = pd.to_numeric(df['lap_duration'], errors='coerce')
    df = df.dropna(subset=['lap_duration', 'compound', 'driver_number'])
    print(f"  Laps after removing NaNs: {len(df)}")

    # Filter out known pit laps if data is available
    if 'is_pit_in_lap' in df.columns and 'is_pit_out_lap' in df.columns:
        initial_count = len(df)
        df = df[~(df['is_pit_in_lap'].astype(bool) | df['is_pit_out_lap'].astype(bool))]
        print(f"  Laps after removing pit in/out: {len(df)} (Removed {initial_count - len(df)})")

    # Filter based on percentiles per group to remove outliers (e.g., SC laps)
    initial_count = len(df)
    print(f"  Applying percentile filter ({LAP_PACE_FILTER_LOWER_PERCENTILE * 100:.0f}% - {LAP_PACE_FILTER_UPPER_PERCENTILE * 100:.0f}%) per group: {group_keys}")

    # Define the filter function
    def percentile_filter(group):
        # Ensure there are enough laps to calculate quantile
        if len(group) < 10: # Arbitrary minimum, adjust if needed
            return group # Don't filter small groups
        lower_bound = group['lap_duration'].quantile(LAP_PACE_FILTER_LOWER_PERCENTILE)
        upper_bound = group['lap_duration'].quantile(LAP_PACE_FILTER_UPPER_PERCENTILE)
        return group[group['lap_duration'].between(lower_bound, upper_bound, inclusive='both')]

    # Apply the filter, handle potential errors during apply
    try:
         # Use group_keys=False to avoid adding group keys as index, which can cause issues
        df_filtered = df.groupby(group_keys, group_keys=False).apply(percentile_filter, include_groups=False)
    except Exception as e:
        print(f"    Error during percentile filtering: {e}. Returning unfiltered data for this group.")
        # In case of error (e.g. incompatible data types), fall back to original df.
        # A more robust approach might involve debugging the specific group causing the error.
        df_filtered = df


    print(f"  Laps after percentile filter: {len(df_filtered)} (Removed {initial_count - len(df_filtered)})")

    return df_filtered

def analyze_compound_specific_performance(practice_lap_data, race_lap_data):
    """
    Analyze the predictive value of FP2 average pace (lap time) on Race average pace,
    focusing on medium and hard tire compounds.

    Args:
        practice_lap_data: List of documents from get_lap_data for FP2.
        race_lap_data: List of documents from get_lap_data for Race sessions.

    Returns:
        Dictionary with overall Spearman correlation results for pace {'MEDIUM': float, 'HARD': float}
    """
    print("\nAnalyzing compound-specific PACE correlation (FP2 vs Race)...")

    practice_df = pd.DataFrame(practice_lap_data)
    race_df = pd.DataFrame(race_lap_data)

    if practice_df.empty or race_df.empty:
        print("Warning: Missing practice or race lap data for pace correlation analysis.")
        return {'MEDIUM': None, 'HARD': None}

    # --- Filter Data ---
    compounds = ['MEDIUM', 'HARD']
    practice_df = practice_df[practice_df['compound'].isin(compounds)]
    race_df = race_df[race_df['compound'].isin(compounds)]

    print("Filtering Practice laps:")
    practice_df_filtered = filter_lap_data_for_pace_analysis(practice_df, ['driver_number', 'compound'])
    print("\nFiltering Race laps:")
    race_df_filtered = filter_lap_data_for_pace_analysis(race_df, ['driver_number', 'compound'])


    if practice_df_filtered.empty or race_df_filtered.empty:
        print("Warning: No valid MEDIUM/HARD lap data found after filtering.")
        return {'MEDIUM': None, 'HARD': None}

    # --- Calculate Average Pace ---
    practice_pace_avg = practice_df_filtered.groupby(['driver_number', 'compound'])['lap_duration'].mean().reset_index()
    race_pace_avg = race_df_filtered.groupby(['driver_number', 'compound'])['lap_duration'].mean().reset_index()

    # --- Merge and Correlate ---
    merged_data = pd.merge(practice_pace_avg, race_pace_avg, on=['driver_number', 'compound'], suffixes=('_practice', '_race'))

    if merged_data.empty:
        print("Warning: No matching driver/compound data between practice and race after averaging.")
        return {'MEDIUM': None, 'HARD': None}

    correlation_results = {}
    print("\nCalculating Overall Pace Correlation:")
    for compound in compounds:
        compound_data = merged_data[merged_data['compound'] == compound]
        if compound_data.shape[0] >= MIN_LAPS_FOR_CORRELATION:
            try:
                # Correlate Practice pace with Race pace
                correlation, p_value = spearmanr(compound_data['lap_duration_practice'], compound_data['lap_duration_race'])
                correlation_results[compound] = correlation
                print(f"  - {compound}: Correlation={correlation:.3f}, p-value={p_value:.3f} (based on {compound_data.shape[0]} drivers)")
            except Exception as e:
                 print(f"  - {compound}: Error calculating correlation - {e}")
                 correlation_results[compound] = None
        else:
            print(f"  - {compound}: Not enough data points ({compound_data.shape[0]} < {MIN_LAPS_FOR_CORRELATION}) for correlation.")
            correlation_results[compound] = None

    return correlation_results

def analyze_compound_specific_performance_by_circuit(practice_lap_data, race_lap_data):
    """
    Analyze the predictive value of FP2 average pace on Race average pace per circuit,
    focusing on medium and hard tire compounds.

    Args:
        practice_lap_data: List of documents from get_lap_data for FP2.
        race_lap_data: List of documents from get_lap_data for Race sessions.

    Returns:
        Dict of circuits mapping to compound-specific pace correlation results.
        {circuit: {'MEDIUM': float, 'HARD': float}}
    """
    print("\nAnalyzing compound-specific PACE correlation per circuit (FP2 vs Race)...")

    practice_df = pd.DataFrame(practice_lap_data)
    race_df = pd.DataFrame(race_lap_data)

    if practice_df.empty or race_df.empty:
        print("Warning: Missing practice or race lap data for circuit-specific pace analysis.")
        return {}

    # --- Filter Data ---
    compounds = ['MEDIUM', 'HARD']
    practice_df = practice_df[practice_df['compound'].isin(compounds)]
    race_df = race_df[race_df['compound'].isin(compounds)]

    # Determine circuit identifier (use 'circuit' if available, else 'race_name')
    circuit_col = 'circuit' if 'circuit' in practice_df.columns and 'circuit' in race_df.columns else 'race_name'

    print("Filtering Practice laps per circuit:")
    practice_df_filtered = filter_lap_data_for_pace_analysis(practice_df, [circuit_col, 'driver_number', 'compound'])
    print("\nFiltering Race laps per circuit:")
    race_df_filtered = filter_lap_data_for_pace_analysis(race_df, [circuit_col, 'driver_number', 'compound'])


    if practice_df_filtered.empty or race_df_filtered.empty:
        print("Warning: No valid MEDIUM/HARD lap data found after filtering for circuit analysis.")
        return {}

    # --- Calculate Average Pace per circuit/driver/compound ---
    practice_pace_avg = practice_df_filtered.groupby([circuit_col, 'driver_number', 'compound'])['lap_duration'].mean().reset_index()
    race_pace_avg = race_df_filtered.groupby([circuit_col, 'driver_number', 'compound'])['lap_duration'].mean().reset_index()

    # --- Merge and Correlate per Circuit ---
    merged = pd.merge(practice_pace_avg, race_pace_avg, on=[circuit_col, 'driver_number', 'compound'], suffixes=('_practice', '_race'))

    if merged.empty:
        print("Warning: No matching circuit/driver/compound data between practice and race after averaging.")
        return {}

    results = {}
    print("\nCalculating Pace Correlation per Circuit:")
    for circuit_name in merged[circuit_col].unique():
        circuit_data = merged[merged[circuit_col] == circuit_name]
        results[circuit_name] = {}
        print(f"  Circuit: {circuit_name}")
        for compound in compounds:
            comp_data = circuit_data[circuit_data['compound'] == compound]
            if comp_data.shape[0] >= MIN_LAPS_FOR_CORRELATION:
                try:
                    corr, p_value = spearmanr(comp_data['lap_duration_practice'], comp_data['lap_duration_race'])
                    results[circuit_name][compound] = corr
                    print(f"    - {compound}: Correlation={corr:.3f}, p-value={p_value:.3f} (based on {comp_data.shape[0]} drivers)")
                except Exception as e:
                    print(f"    - {compound}: Error calculating correlation - {e}")
                    results[circuit_name][compound] = None
            else:
                print(f"    - {compound}: Not enough data points ({comp_data.shape[0]} < {MIN_LAPS_FOR_CORRELATION}) for correlation.")
                results[circuit_name][compound] = None
    return results


# --- Saving and Printing Functions (Mostly Unchanged, check filenames/content) ---

def save_results_to_json(data, filename, output_dir=OUTPUT_DIR):
    """ Generic function to save dictionary data to JSON """
    print(f"Saving data to JSON: {filename} in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, 'w') as f:
             # Custom handler for NaN -> null and other potential non-serializable types
             def json_encoder(obj):
                 if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                     np.int16, np.int32, np.int64, np.uint8,
                                     np.uint16, np.uint32, np.uint64)):
                     return int(obj)
                 elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                     # Convert NaN or infinity to None (JSON null)
                     return None if pd.isna(obj) or np.isinf(obj) else float(obj)
                 elif isinstance(obj, (np.ndarray,)): # Handle numpy arrays
                     return obj.tolist()
                 elif pd.isna(obj): # Handle pandas NaT or other NaNs
                     return None
                 raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

             json.dump(data, f, indent=4, default=json_encoder)
        print(f"Data saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        traceback.print_exc()


def save_stint_percentiles_to_csv(results, filename="stint_tire_life_percentiles.csv", output_dir=OUTPUT_DIR):
    """ Saves STINT tire life percentile analysis to CSV. """
    print(f"Saving stint percentile results to CSV: {filename} in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    csv_data = []
    if "percentiles" in results:
        for race, percentiles_list in results["percentiles"].items():
            for p_info in percentiles_list:
                csv_data.append({
                    "Race": race,
                    "Compound": p_info.get('compound', 'N/A'),
                    "40th Percentile Laps": p_info.get('40th_percentile_laps'),
                    "80th Percentile Laps": p_info.get('80th_percentile_laps')
                })
    if csv_data:
        df = pd.DataFrame(csv_data)
        try:
            df.to_csv(filepath, index=False, float_format='%.1f')
            print(f"Stint percentile results saved to {filepath}")
        except Exception as e: print(f"Error saving stint percentiles to CSV: {e}")
    else: print("No stint percentile data found to save to CSV.")

def save_stint_sequences_to_csv(sequences, filename="stint_tire_sequences.csv", output_dir=OUTPUT_DIR):
    """ Saves most common STINT tire sequences to CSV. """
    print(f"Saving stint tire sequences to CSV: {filename} in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    csv_data = [{"Race": race, "Most Common Stint Sequence": " > ".join(seq)}
                for race, seq in sequences.items() if seq]
    if csv_data:
        df = pd.DataFrame(csv_data)
        try:
            df.to_csv(filepath, index=False)
            print(f"Stint sequences saved to {filepath}")
        except Exception as e: print(f"Error saving stint sequences to CSV: {e}")
    else: print("No stint sequence data to save to CSV.")


def print_summary_results(strategy_results, sequence_results, correlation_results, track_corr_results):
    """ Prints a summary of all analysis results to the console. """
    print("\n" + "="*40)
    print(" F1 Analysis Summary (2024)")
    print("="*40)

    # --- Stint-Based Results ---
    print("\n--- Most Common Starting Tire (Stints) ---")
    if strategy_results.get("strategy"):
        for race, strategy in sorted(strategy_results["strategy"].items()): print(f"{race}: {strategy}")
    else: print("No starting strategy data.")

    print("\n--- Tire Life Percentiles (Stints, Laps) ---")
    if strategy_results.get("percentiles"):
        for race, p_list in sorted(strategy_results["percentiles"].items()):
            print(f"\n{race}:")
            if p_list:
                sorted_p = sorted(p_list, key=lambda x: x.get('compound', ''))
                for p_info in sorted_p:
                    comp = p_info.get('compound', 'N/A')
                    p40 = p_info.get('40th_percentile_laps')
                    p80 = p_info.get('80th_percentile_laps')
                    p40s = f"{p40:.1f}" if p40 is not None and not pd.isna(p40) else "N/A"
                    p80s = f"{p80:.1f}" if p80 is not None and not pd.isna(p80) else "N/A"
                    print(f"  {comp:<7}: 40th={p40s:>5} laps, 80th={p80s:>5} laps")
            else: print("  No percentile data for this race.")
    else: print("No tire life percentile data.")

    print("\n--- Most Common Tire Sequence (Stints) ---")
    if sequence_results:
        for race, seq in sorted(sequence_results.items()): print(f"{race}: {' > '.join(seq) if seq else 'N/A'}")
    else: print("No sequence data.")

    # --- Pace-Based Results ---
    print("\n" + "="*40)
    print(" Pace Correlation Analysis (FP2 vs Race)")
    print("="*40)
    print(f"(Based on Avg Lap Time, Filter: {LAP_PACE_FILTER_LOWER_PERCENTILE*100:.0f}-{LAP_PACE_FILTER_UPPER_PERCENTILE*100:.0f} Percentile)")


    print("\n--- Overall Pace Correlation (MEDIUM/HARD) ---")
    if correlation_results:
        has_results = False
        for compound, corr in correlation_results.items():
             corr_str = f"{corr:.3f}" if corr is not None else "N/A"
             print(f"  {compound}: {corr_str}")
             if corr is not None: has_results = True
        if not has_results: print("  No overall correlation results calculated.")
    else: print("  No overall correlation data available.")

    print("\n--- Per-Circuit Pace Correlation (MEDIUM/HARD) ---")
    if track_corr_results:
        has_results = False
        for circuit, comp_dict in sorted(track_corr_results.items()):
            print(f"\n  {circuit}:")
            circuit_has_results = False
            if comp_dict:
                 for comp, corr in sorted(comp_dict.items()):
                     corr_str = f"{corr:.3f}" if corr is not None else "N/A"
                     print(f"    {comp:<7}: {corr_str}")
                     if corr is not None: circuit_has_results = True
            if not circuit_has_results: print("    No correlation results calculated for this circuit.")
            if circuit_has_results : has_results = True
        if not has_results: print("\n  No per-circuit correlation results calculated.")
    else: print("  No per-circuit correlation data available.")

    print("\n" + "="*40)


def main():
    """Main execution function."""
    print("Starting F1 Pace and Tire Strategy Analysis Script...")
    print("="*50)

    db = None
    try:
        # Connect to MongoDB
        db = connect_to_mongodb()

        # --- Optional Debugging ---
        print_all_2024_session_names(db)
        print("\nCollections in database:", db.list_collection_names())
        print("-"*50)
        # --- End Debugging ---


        # --- Fetch Data ---
        print("--- Fetching Stint Data ---")
        race_stint_data = get_stint_data(db, session_type="Race")

        print("\n--- Fetching Lap Data ---")
        # Fetch Race laps
        race_lap_data = get_lap_data(db, session_type="Race")

        # Fetch FP2 laps (try common names)
        fp2_lap_data = get_lap_data(db, session_type="Practice", session_name="Practice 2")
        if not fp2_lap_data:
             print("Warning: No 'Practice 2' lap data found. Trying 'Free Practice 2'...")
             fp2_lap_data = get_lap_data(db, session_type="Practice", session_name="Free Practice 2")
        if not fp2_lap_data:
             print("Warning: No FP2 lap data found. Pace correlation analysis requires FP2 data.")
             fp2_lap_data = [] # Ensure it's an empty list

        print("-"*50)


        # --- Analyze Data ---
        print("--- Running Analysis ---")
        # Stint-based analysis
        strategy_results = analyze_tire_strategies(race_stint_data)
        sequence_results = analyze_tire_sequences(race_stint_data)

        # Pace-based analysis (using lap data)
        pace_correlation_overall = analyze_compound_specific_performance(fp2_lap_data, race_lap_data)
        pace_correlation_by_circuit = analyze_compound_specific_performance_by_circuit(fp2_lap_data, race_lap_data)

        print("-"*50)


        # --- Save Results ---
        print("--- Saving Results ---")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save Stint analysis results
        save_results_to_json(strategy_results, "stint_strategy_analysis.json", output_dir=OUTPUT_DIR)
        save_stint_percentiles_to_csv(strategy_results, output_dir=OUTPUT_DIR)
        save_stint_sequences_to_csv(sequence_results, output_dir=OUTPUT_DIR)

        # Save Pace correlation results
        save_results_to_json(pace_correlation_overall, "pace_correlation_overall.json", output_dir=OUTPUT_DIR)
        save_results_to_json(pace_correlation_by_circuit, "pace_correlation_by_circuit.json", output_dir=OUTPUT_DIR)

        print("-"*50)


        # --- Print Summary ---
        print_summary_results(strategy_results, sequence_results, pace_correlation_overall, pace_correlation_by_circuit)

        print("\nAnalysis complete!")

    except ImportError as e:
         print(f"Import Error: {e}. Please ensure pandas, pymongo, scipy, numpy are installed (`pip install pandas pymongo scipy numpy`)")
    except ConnectionError as e:
        print(f"MongoDB Connection Error: {e}")
        print(f"Ensure MongoDB is running and accessible via '{MONGO_CONNECTION_STRING}'.")
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("\n--- Traceback ---"); traceback.print_exc(); print("-----------------")
    finally:
         if db is not None and hasattr(db, 'client') and db.client:
             print("Closing MongoDB connection.")
             db.client.close()


if __name__ == "__main__":
    main()