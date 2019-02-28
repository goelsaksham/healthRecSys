DROP TABLE sleep_cycles;
DROP TABLE sleep_summary;
DROP TABLE sleep_intraday_data;
DROP TABLE activity_intraday_data;
DROP TABLE activity_type_summary;
DROP TABLE heart_rate_intraday_data;
DROP TABLE raw_device_data;

CREATE TABLE sleep_cycles (
        subject_id character varying(10) NOT NULL references subject(id),
        efficiency float,
        isMainSleep integer,
	minutesAfterWakeup float,
        minutesAsleep float,
        minutesAwake float,
        minutesToFallAsleep float,
        startTime timestamp,
        duration float
);

CREATE TABLE sleep_summary (
        subject_id character varying(10) NOT NULL references subject(id),
        sleep_date date NOT NULL,
        totalMinutesAsleep float,
        totalSleepRecords integer,
        totalTimeInBed float
);

CREATE TABLE sleep_intraday_data (
        subject_id character varying(10) NOT NULL references subject(id),
        start_time timestamp NOT NULL,
        time_sec integer,
        sleep_type integer
);

CREATE TABLE activity_intraday_data (
        subject_id character varying(10) NOT NULL references subject(id),
        activity_date date NOT NULL,
        calories float,
        activity_level integer,
	activity_level_desc character varying(255),
	activity_met float
);

CREATE TABLE activity_type_summary (
        subject_id character varying(10) NOT NULL references subject(id),
        activity_level integer,
	value float NOT NULL,
	date date NOT NULL
);

CREATE TABLE heart_rate_intraday_data (
        subject_id character varying(10) NOT NULL references subject(id),
        log_time timestamp,
        heart_rate integer
);

CREATE TABLE raw_device_data (
   	subject_id character varying(10) NOT NULL references subject(id),
   	data jsonb,
   	pull_time timestamp without time zone,
   	first_date date NOT NULL,
   	wearable_data_type bigint NOT NULL references wearable_data_type(id)
);