#!/usr/bin/env python3
"""
F1 Tire Analysis Script

This script connects to the MongoDB database, executes queries to analyze tire usage
data for all races in the 2024 Formula 1 season, and formats the results in a readable way.
"""

import os
import argparse
import pandas as pd
from pymongo import MongoClient
from tabulate import tabulate
import matplotlib.pyplot as plt
import json
from datetime import datetime

# MongoDB connection settings
MONGO_CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
DB_NAME = "openf1-livetiming"

def connect_to_mongodb():
    """Connect to MongoDB and return the database object."""
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    return db

def get_tire_data_for_2024_races(db):
    """
    Execute the MongoDB aggregation pipeline to get tire data for all 2024 races.
    
    Args:
        db: MongoDB database object
        
    Returns:
        List of documents with tire data
    """
    pipeline = [
        # Step 1: Join with sessions to filter only race sessions in 2024
        {
            "$lookup": {
                "from": "sessions",
                "localField": "session_key",
                "foreignField": "session_key",
                "as": "session"
            }
        },
        {
            "$unwind": "$session"
        },
        {
            "$match": {
                "session.year": 2024,
                "session.session_type": "Race"
            }
        },
        
        # Step 2: Join with meetings to get race information
        {
            "$lookup": {
                "from": "meetings",
                "localField": "meeting_key",
                "foreignField": "meeting_key",
                "as": "meeting"
            }
        },
        {
            "$unwind": "$meeting"
        },
        
        # Step 3: Join with drivers to get driver information
        {
            "$lookup": {
                "from": "drivers",
                "let": { 
                    "session_key": "$session_key", 
                    "driver_number": "$driver_number" 
                },
                "pipeline": [
                    {
                        "$match": {
                            "$expr": { 
                                "$and": [
                                    { "$eq": ["$session_key", "$$session_key"] },
                                    { "$eq": ["$driver_number", "$$driver_number"] }
                                ]
                            }
                        }
                    }
                ],
                "as": "driver"
            }
        },
        {
            "$unwind": "$driver"
        },
        
        # Step 4: Calculate laps per stint and format output
        {
            "$project": {
                "race_name": "$meeting.meeting_name",
                "race_date": "$session.date_start",
                "circuit": "$meeting.circuit_short_name",
                "driver_name": "$driver.full_name",
                "driver_number": "$driver_number",
                "team_name": "$driver.team_name",
                "stint_number": "$stint_number",
                "compound": "$compound",
                "lap_start": "$lap_start",
                "lap_end": "$lap_end",
                "laps_on_tire": { 
                    "$cond": {
                        "if": { "$and": [{ "$ne": ["$lap_start", None] }, { "$ne": ["$lap_end", None] }] },
                        "then": { "$add": [{ "$subtract": ["$lap_end", "$lap_start"] }, 1] },  # +1 because lap_start and lap_end are inclusive
                        "else": None
                    }
                },
                "tyre_age_at_start": "$tyre_age_at_start"
            }
        },
        
        # Step 5: Sort by race date, driver number, and stint number
        {
            "$sort": {
                "race_date": 1,
                "driver_number": 1,
                "stint_number": 1
            }
        }
    ]
    
    results = list(db.stints.aggregate(pipeline))
    return results

def format_results_by_race(results):
    """
    Format the results grouped by race.
    
    Args:
        results: List of documents with tire data
        
    Returns:
        Dictionary with race names as keys and formatted data as values
    """
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)
    
    if df.empty:
        print("No data found for 2024 races.")
        return {}
    
    # Convert race_date to datetime if it's not already
    if 'race_date' in df.columns:
        df['race_date'] = pd.to_datetime(df['race_date'])
    
    # Group by race
    races = {}
    for race_name, race_df in df.groupby('race_name'):
        race_date = race_df['race_date'].iloc[0]
        circuit = race_df['circuit'].iloc[0]
        
        # Format race header
        race_header = f"{race_name} ({circuit}) - {race_date.strftime('%Y-%m-%d')}"
        
        # Group by driver
        driver_data = []
        for (driver_name, driver_number), driver_df in race_df.groupby(['driver_name', 'driver_number']):
            team_name = driver_df['team_name'].iloc[0]
            
            # Format driver stints
            stints = []
            for _, stint in driver_df.iterrows():
                compound = stint['compound'] if stint['compound'] else 'Unknown'
                lap_start = stint['lap_start'] if pd.notna(stint['lap_start']) else 'N/A'
                lap_end = stint['lap_end'] if pd.notna(stint['lap_end']) else 'N/A'
                laps_on_tire = stint['laps_on_tire'] if pd.notna(stint['laps_on_tire']) else 'N/A'
                
                stints.append({
                    'stint_number': stint['stint_number'],
                    'compound': compound,
                    'lap_start': lap_start,
                    'lap_end': lap_end,
                    'laps_on_tire': laps_on_tire,
                    'tyre_age_at_start': stint['tyre_age_at_start'] if pd.notna(stint['tyre_age_at_start']) else 0
                })
            
            driver_data.append({
                'driver_name': driver_name,
                'driver_number': driver_number,
                'team_name': team_name,
                'stints': stints
            })
        
        races[race_header] = driver_data
    
    return races

def print_results(races):
    """
    Print the formatted results to the console.
    
    Args:
        races: Dictionary with race names as keys and formatted data as values
    """
    if not races:
        print("No data to display.")
        return
    
    for race_header, driver_data in races.items():
        print("\n" + "=" * 80)
        print(f"## {race_header}")
        print("=" * 80)
        
        for driver in driver_data:
            print(f"\nDriver: {driver['driver_name']} (#{driver['driver_number']}) - {driver['team_name']}")
            
            # Create a table for the stints
            stint_data = []
            for stint in driver['stints']:
                stint_data.append([
                    stint['stint_number'],
                    stint['compound'],
                    stint['lap_start'],
                    stint['lap_end'],
                    stint['laps_on_tire'],
                    stint['tyre_age_at_start']
                ])
            
            headers = ["Stint", "Compound", "Lap Start", "Lap End", "Laps on Tire", "Tire Age at Start"]
            print(tabulate(stint_data, headers=headers, tablefmt="grid"))

def export_to_csv(races, output_dir="output/2024 tyre stints"):
    """
    Export the results to CSV files, one per race.
    
    Args:
        races: Dictionary with race names as keys and formatted data as values
        output_dir: Directory to save the CSV files
    """
    if not races:
        print("No data to export.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for race_header, driver_data in races.items():
        # Create a clean filename from the race header
        filename = race_header.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        filename = ''.join(c for c in filename if c.isalnum() or c == '_')
        filename = f"{filename}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Flatten the data for CSV export
        flat_data = []
        for driver in driver_data:
            for stint in driver['stints']:
                flat_data.append({
                    'race': race_header,
                    'driver_name': driver['driver_name'],
                    'driver_number': driver['driver_number'],
                    'team_name': driver['team_name'],
                    'stint_number': stint['stint_number'],
                    'compound': stint['compound'],
                    'lap_start': stint['lap_start'],
                    'lap_end': stint['lap_end'],
                    'laps_on_tire': stint['laps_on_tire'],
                    'tyre_age_at_start': stint['tyre_age_at_start']
                })
        
        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(flat_data)
        df.to_csv(filepath, index=False)
        print(f"Exported data for {race_header} to {filepath}")

def create_visualizations(races, output_dir="output/2024 tyre stints"):
    """
    Create visualizations of the tire data.
    
    Args:
        races: Dictionary with race names as keys and formatted data as values
        output_dir: Directory to save the visualizations
    """
    if not races:
        print("No data to visualize.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for race_header, driver_data in races.items():
        # Create a clean filename from the race header
        filename = race_header.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        filename = ''.join(c for c in filename if c.isalnum() or c == '_')
        
        # Create a bar chart of laps on each tire compound by driver
        plt.figure(figsize=(15, 10))
        
        # Prepare data for plotting
        drivers = []
        compounds = set()
        driver_compound_laps = {}
        
        for driver in driver_data:
            driver_name = f"{driver['driver_name']} (#{driver['driver_number']})"
            drivers.append(driver_name)
            driver_compound_laps[driver_name] = {}
            
            for stint in driver['stints']:
                compound = stint['compound']
                laps = stint['laps_on_tire']
                
                if compound and laps != 'N/A':
                    compounds.add(compound)
                    if compound in driver_compound_laps[driver_name]:
                        driver_compound_laps[driver_name][compound] += laps
                    else:
                        driver_compound_laps[driver_name][compound] = laps
        
        # Convert compounds to a sorted list
        compounds = sorted(list(compounds))
        
        # Create the plot
        bar_width = 0.8 / len(compounds)
        index = range(len(drivers))
        
        for i, compound in enumerate(compounds):
            values = [driver_compound_laps.get(driver, {}).get(compound, 0) for driver in drivers]
            bars = plt.bar([x + i * bar_width for x in index], values, bar_width, label=compound)
            # Add labels to the bars
            for bar in bars:
                yval = bar.get_height()
                if yval > 0: # Only add label if the number of laps is greater than 0
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', fontsize=8) # va: vertical alignment, ha: horizontal alignment
        
        plt.xlabel('Driver')
        plt.ylabel('Laps')
        plt.title(f'Laps on Each Tire Compound - {race_header}')
        plt.xticks([x + (len(compounds) - 1) * bar_width / 2 for x in index], drivers, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{filename}_tire_usage.png"))
        plt.close()
        
        print(f"Created visualization for {race_header}")

def main():
    """Main function to execute the script."""
    parser = argparse.ArgumentParser(description="F1 Tire Analysis Script")
    parser.add_argument(
        "--analysis_type",
        type=str,
        default="stints",
        choices=["stints", "strategy"],
        help="Type of analysis to perform: 'stints' for per-driver stints, 'strategy' for tire strategy summary."
    )
    args = parser.parse_args()
    print(f"Parsed analysis_type argument: {args.analysis_type}")

    print("F1 Tire Analysis Script")
    print("======================")
    
    # Connect to MongoDB
    print("\nConnecting to MongoDB...")
    try:
        db = connect_to_mongodb()
        print("Connected successfully!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return
    
    # Get tire data for 2024 races
    print("\nFetching tire data for 2024 races...")
    try:
        results = get_tire_data_for_2024_races(db)
        print(f"Found {len(results)} stint records.")
    except Exception as e:
        print(f"Error fetching tire data: {e}")
        return
    
    # Format results by race
    print("\nFormatting results by race...")
    races = format_results_by_race(results)
    
    # Print results
    print("\nPrinting results:")
    print_results(races)
    
    # Export to CSV
    print("\nExporting results to CSV...")
    export_to_csv(races)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(races)
    
    print("\nAnalysis complete!")

def analyze_tire_strategies(db):
    """
    Analyze tire strategies for 2024 races, including most common starting tire
    and percentiles for laps on each compound.

    Args:
        db: MongoDB database object

    Returns:
        Dictionary with tire strategy analysis results
    """
    pipeline = [
        # Step 1: Join with sessions to filter only race sessions in 2024
        {
            "$lookup": {
                "from": "sessions",
                "localField": "session_key",
                "foreignField": "session_key",
                "as": "session"
            }
        },
        {
            "$unwind": "$session"
        },
        {
            "$match": {
                "session.year": 2024,
                "session.session_type": "Race"
            }
        },
        # Step 2: Join with meetings to get race information
        {
            "$lookup": {
                "from": "meetings",
                "localField": "meeting_key",
                "foreignField": "meeting_key",
                "as": "meeting"
            }
        },
        {
            "$unwind": "$meeting"
        },
        # Step 3: Project necessary fields for analysis
        {
            "$project": {
                "_id": 0,
                "race_name": "$meeting.meeting_name",
                "driver_number": "$driver_number",
                "compound": "$compound",
                "lap_start": "$lap_start",
                "lap_end": "$lap_end",
                "laps_on_tire": {
                    "$cond": {
                        "if": { "$and": [{ "$ne": ["$lap_start", None] }, { "$ne": ["$lap_end", None] }] },
                        "then": { "$add": [{ "$subtract": ["$lap_end", "$lap_start"] }, 1] },
                        "else": None
                    }
                }
            }
        }
    ]

    results = list(db.stints.aggregate(pipeline))
    # Process results to determine most common strategy and calculate percentiles
    df = pd.DataFrame(results)

    if df.empty:
        print("No data found for 2024 races.")
        return {"strategy": {}, "percentiles": {}}

    # Calculate most common starting tire strategy
    starting_stints = df.loc[df.groupby(['race_name', 'driver_number'])['lap_start'].idxmin()]
    most_common_strategy = starting_stints.groupby('race_name')['compound'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').to_dict()

    # Calculate percentiles for laps on each tire compound per race
    percentiles_data = df.groupby(['race_name', 'compound'])['laps_on_tire'].quantile([0.4, 0.8]).unstack().reset_index()
    percentiles_data.columns = ['race_name', 'compound', '40th_percentile_laps', '80th_percentile_laps']

    percentile_analysis = percentiles_data.groupby('race_name').apply(lambda x: x[['compound', '40th_percentile_laps', '80th_percentile_laps']].to_dict('records')).to_dict()

    return {
        "strategy": most_common_strategy,
        "percentiles": percentile_analysis
    }

    if args.analysis_type == "stints":
        # Get tire data for 2024 races
        print("\nFetching tire data for 2024 races...")
        try:
            results = get_tire_data_for_2024_races(db)
            print(f"Found {len(results)} stint records.")
        except Exception as e:
            print(f"Error fetching tire data: {e}")
            return

        # Format results by race
        print("\nFormatting results by race...")
        races = format_results_by_race(results)

        # Print results
        print("\nPrinting results:")
        print_results(races)

        # Export to CSV
        print("\nExporting results to CSV...")
        export_to_csv(races)

        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(races)

        print("\nAnalysis complete!")

    elif args.analysis_type.strip().lower() == "strategy":
        # Analyze tire strategies and percentiles
        print("\nAnalyzing tire strategies and percentiles...")
        try:
            strategy_percentile_results = analyze_tire_strategies(db)
            print("\nTire Strategy Analysis:")
            print("=======================")
            for race, strategy in strategy_percentile_results["strategy"].items():
                print(f"Most common starting tire for {race}: {strategy}")

            print("\nTyre Laps Percentile Analysis:")
            print("==============================")
            for race, percentiles in strategy_percentile_results["percentiles"].items():
                print(f"\n{race}:")
                for percentile_info in percentiles:
                    print(f"  Compound: {percentile_info['compound']}, 40th Percentile Laps: {percentile_info['40th_percentile_laps']:.2f}, 80th Percentile Laps: {percentile_info['80th_percentile_laps']:.2f}")
    
            # Save strategy analysis results to a JSON file
            print("\nSaving tire strategy analysis to JSON...")
            output_dir = "output/2024 tyre strategy"
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, "tire_strategy_analysis.json")
            with open(filepath, 'w') as f:
                json.dump(strategy_percentile_results, f, indent=4)
            print(f"Tire strategy analysis results saved to {filepath}")

        except Exception as e:
            print(f"An error occurred during tire strategy analysis or saving: {e}")
            import traceback
            traceback.print_exc()

    print("\nScript execution finished.")


if __name__ == "__main__":
    main()