class MATSimWriter:
    def __init__(self, config: Config):
        self.config = config
        self.df = self.load_population_data()

    def load_population_data(self):
        """
        Load population data (expects it in the output folder).
        """
        population_file = self.config.get("matsim_writer.input.population_df")
        if not os.path.exists(population_file):
            logger.error(f"Expected population data file not found: {population_file}")
            sys.exit(1)
        return pd.read_csv(population_file)

    def write_plans_to_matsim_xml(self):
        """
        Writes plans to MATSim XML format.
        """
        logger.info("Writing plans to MATSim xml...")
        output_file = config.get("matsim_writer.output.plans")

        with open(output_file, 'wb+') as f_write:
            writer = matsim.writers.PopulationWriter(f_write)
            writer.start_population()

            for _, group in self.df.groupby([s.UNIQUE_P_ID_COL]):
                writer.start_person(group[s.UNIQUE_P_ID_COL].iloc[0])
                writer.start_plan(selected=True)

                writer.add_activity(
                    type="home",
                    x=group[s.HOME_LOC_COL].iloc[0].x,
                    y=group[s.HOME_LOC_COL].iloc[0].y,
                    end_time=abs(h.seconds_from_datetime(group[s.LEG_START_TIME_COL].iloc[0]))
                )

                for idx, row in group.iterrows():
                    writer.add_leg(mode=row[s.MODE_TRANSLATED_STRING_COL])
                    max_dur = round(row.get(s.ACT_DUR_SECONDS_COL, 3600) / 600) * 600
                    writer.add_activity(
                        type=f"{row[s.ACTIVITY_TRANSLATED_STRING_COL]}_{max_dur}",
                        x=row[s.COORD_TO_COL].x,
                        y=row[s.COORD_TO_COL].y,
                        start_time=max_dur
                    )
                writer.end_plan()
                writer.end_person()
            writer.end_population()

        logger.info(f"Wrote plans to MATSim xml: {output_file}")
        return output_file

    def write_households_to_matsim_xml(self):
        """
        Writes households to MATSim XML format.
        """
        logger.info("Writing households to MATSim xml...")
        output_file = config.get("matsim_writer.output.households")

        with open(output_file, 'wb+') as f_write:
            households_writer = matsim.writers.HouseholdsWriter(f_write)
            households_writer.start_households()

            for _, hh in self.df.groupby([s.UNIQUE_HH_ID_COL]):
                household_id = hh[s.UNIQUE_HH_ID_COL].iloc[0]
                person_ids = hh[s.UNIQUE_P_ID_COL].unique().tolist()
                households_writer.start_household(household_id)
                households_writer.add_members(person_ids)
                households_writer.end_household()

            households_writer.end_households()
        logger.info(f"Wrote households to MATSim xml: {output_file}")

    def write_facilities_to_matsim_xml(self):
        """
        Writes facilities data to MATSim XML format.
        """
        logger.info("Writing facilities to MATSim xml...")
        output_file = config.get("matsim_writer.output.facilities")

        with open(output_file, 'wb+') as f_write:
            facilities_writer = matsim.writers.FacilitiesWriter(f_write)
            facilities_writer.start_facilities()

            for row in self.df.itertuples():
                facility_id = getattr(row, s.FACILITY_ID_COL)
                x = getattr(row, s.FACILITY_X_COL)
                y = getattr(row, s.FACILITY_Y_COL)
                activities = list(getattr(row, s.FACILITY_ACTIVITIES_COL))

                facilities_writer.start_facility(facility_id, x, y)
                for activity in activities:
                    facilities_writer.add_activity(activity)
                facilities_writer.end_facility()

            facilities_writer.end_facilities()
        logger.info(f"Wrote facilities to MATSim xml: {output_file}")

    def write_vehicles_to_matsim_xml(self):
        """
        Writes vehicles data to MATSim XML format.
        """
        logger.info("Writing vehicles to MATSim xml...")
        output_file = config.get("matsim_writer.output.vehicles")

        with open(output_file, 'wb+') as f_write:
            vehicle_writer = matsim.writers.VehiclesWriter(f_write)
            vehicle_writer.start_vehicle_definitions()

            vehicle_writer.add_vehicle_type(
                vehicle_id="car",
                length=7.5, width=1.0, pce=1.0,
                network_mode="car"
            )

            for _, hh in self.df.groupby([s.UNIQUE_HH_ID_COL]):
                vehicle_ids = hh.get(s.LIST_OF_CARS_COL, []).iloc[0]
                if vehicle_ids:
                    for vehicle_id in vehicle_ids:
                        vehicle_writer.add_vehicle(vehicle_id=vehicle_id, vehicle_type="car")

            vehicle_writer.end_vehicle_definitions()
        logger.info(f"Wrote vehicles to MATSim xml: {output_file}")
