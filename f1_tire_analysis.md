# F1 Tire Analysis

This document outlines MongoDB queries and a Python script for analyzing tire usage data for all races in the 2024 Formula 1 season.

## MongoDB Queries

### Query 1: Tire Usage Analysis for All 2024 Races

```javascript
// Find all stints for races in 2024
db.stints.aggregate([
  // Step 1: Join with sessions to filter only race sessions in 2024
  {
    $lookup: {
      from: "sessions",
      localField: "session_key",
      foreignField: "session_key",
      as: "session"
    }
  },
  {
    $unwind: "$session"
  },
  {
    $match: {
      "session.year": 2024,
      "session.session_type": "Race"
    }
  },
  
  // Step 2: Join with meetings to get race information
  {
    $lookup: {
      from: "meetings",
      localField: "meeting_key",
      foreignField: "meeting_key",
      as: "meeting"
    }
  },
  {
    $unwind: "$meeting"
  },
  
  // Step 3: Join with drivers to get driver information
  {
    $lookup: {
      from: "drivers",
      let: { 
        session_key: "$session_key", 
        driver_number: "$driver_number" 
      },
      pipeline: [
        {
          $match: {
            $expr: { 
              $and: [
                { $eq: ["$session_key", "$$session_key"] },
                { $eq: ["$driver_number", "$$driver_number"] }
              ]
            }
          }
        }
      ],
      as: "driver"
    }
  },
  {
    $unwind: "$driver"
  },
  
  // Step 4: Calculate laps per stint and format output
  {
    $project: {
      race_name: "$meeting.meeting_name",
      race_date: "$session.date_start",
      circuit: "$meeting.circuit_short_name",
      driver_name: "$driver.full_name",
      driver_number: "$driver_number",
      team_name: "$driver.team_name",
      stint_number: "$stint_number",
      compound: "$compound",
      lap_start: "$lap_start",
      lap_end: "$lap_end",
      laps_on_tire: { 
        $cond: {
          if: { $and: [{ $ne: ["$lap_start", null] }, { $ne: ["$lap_end", null] }] },
          then: { $add: [{ $subtract: ["$lap_end", "$lap_start"] }, 1] },  // +1 because lap_start and lap_end are inclusive
          else: null
        }
      },
      tyre_age_at_start: "$tyre_age_at_start"
    }
  },
  
  // Step 5: Sort by race date, driver number, and stint number
  {
    $sort: {
      "race_date": 1,
      "driver_number": 1,
      "stint_number": 1
    }
  }
])
```

## Python Script Implementation

Below is a Python script that connects to the MongoDB database, executes the query, and formats the results in a readable way:

```python
#!/usr/bin/env python3
"""
F1 Tire Analysis Script

This script connects to the MongoDB database, executes queries to analyze tire usage
data for all races in the 2024 Formula 1 season, and formats the results in a readable way.
"""

import os
import pandas as pd
from pymongo import MongoClient
from tabulate import tabulate
import matplotlib.pyplot as plt
from datetime import datetime

# MongoDB connection settings
MONGO_CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
DB_NAME = "openf1"

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

def export_to_csv(races, output_dir="output"):
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

def create_visualizations(races, output_dir="output"):
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
            plt.bar([x + i * bar_width for x in index], values, bar_width, label=compound)
        
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

if __name__ == "__main__":
    main()
```

## Usage Instructions

1. Ensure you have MongoDB installed and running with the OpenF1 database populated.
2. Set the `MONGO_CONNECTION_STRING` environment variable to connect to your MongoDB instance:
   ```bash
   export MONGO_CONNECTION_STRING="mongodb://localhost:27017"
   ```
3. Install the required Python packages:
   ```bash
   pip install pymongo pandas tabulate matplotlib
   ```
4. Run the script:
   ```bash
   python f1_tire_analysis.py
   ```

## Output

The script will:
1. Connect to the MongoDB database
2. Execute the query to get tire data for all 2024 races
3. Format the results grouped by race
4. Print the results to the console in a tabular format
5. Export the results to CSV files (one per race) in the `output` directory
6. Create visualizations of the tire data in the `output` directory

## Data Structure

For each race, the script will show:
- Race name, circuit, and date
- For each driver:
  - Driver name, number, and team
  - For each stint:
    - Stint number
    - Tire compound
    - Lap start and end
    - Number of laps on the tire
    - Tire age at the start of the stint

## Visualizations

The script will create bar charts showing the number of laps each driver spent on each tire compound for each race.