CREATE TABLE activity_level (
   	id character varying(10) NOT NULL references subject(id),
   	name character varying(255) DEFAULT NULL::character varying,
   	description character varying(255) DEFAULT NULL::character varying
);

CREATE TABLE raw_device_data (
   	subject_id character varying(10) NOT NULL references subject(id),
   	wearable_data_type bigint NOT NULL references wearable_data_type(id),
   	data jsonb,
   	pull_time timestamp without time zone,
   	first_date date
);

CREATE TABLE subject (
   	id character varying(10) NOT NULL primary key,
   	birth_year smallint,
   	gender character varying(10),
   	provider character varying(255),
   	timezone character varying(255),
   	user_name character varying(255) NOT NULL,
);

CREATE TABLE sync_rows (
   	id bigint NOT NULL,
   	created_at timestamp without time zone NOT NULL,
   	updated_at timestamp without time zone NOT NULL,
   	last_row_id bigint,
   	table_name character varying(255)
);

CREATE TABLE system_user (
   	id bigint NOT NULL,
   	user_name character varying(50),
   	password character varying(50),
   	email character varying(50),
   	enabled boolean,
   	user_role smallint,
);

CREATE TABLE userconnection (
   	id integer NOT NULL,
   	accesstoken character varying(1024),
   	displayname character varying(255),
   	expiretime bigint,
   	imageurl character varying(255),
   	profileurl character varying(255),
   	providerid character varying(255),
   	provideruserid character varying(255),
   	rank integer,
   	refreshtoken character varying(255),
   	secret character varying(255),
   	userid character varying(255)
);

CREATE TABLE wearable_data_type (
   	id bigint NOT NULL primary key,
   	description character varying(255),
   	inactive_value integer,
   	name character varying(255)
);

CREATE TABLE activity_intraday_data (
        subject_id character varying(10) NOT NULL references subject(id),
        activity_date_time timestamp NOT NULL,
        calories float,
        activity_level integer,
	      activity_level_desc character varying(255),
	      activity_met float
);

CREATE TABLE activity_type_summary (
        subject_id character varying(10) NOT NULL references subject(id),
        date date NOT NULL,
        activity_level integer,
	      value float NOT NULL
);


CREATE TABLE sleep_intraday_data (
        subject_id character varying(10) NOT NULL references subject(id),
        start_time timestamp NOT NULL,
        time_sec integer,
        sleep_type integer
);

CREATE TABLE sleep_type_summary (
        subject_id character varying(10) NOT NULL references subject(id),
        sleep_date date NOT NULL,
        deep_count float,
        deep_minutes float,
        deep_thirtydayavgmin float,
        rem_count float,
        rem_minutes float,
        rem_thirtydayavgmin float,
        light_count float,
        light_minutes float,
        light_thirtydayavgmin float,
        wake_count float,
        wake_minutes float,
        wake_thirtydayavgmin float
);

CREATE TABLE sleep_cycles (
        subject_id character varying(10) NOT NULL references subject(id),
        duration float,
        efficiency float,
        isMainSleep integer,
        minutesAfterWakeup float,
        minutesAsleep float,
        minutesAwake float,
        minutesToFallAsleep float,
        startTime timestamp,
        timeInBed float
);

CREATE TABLE sleep_summary (
        subject_id character varying(10) NOT NULL references subject(id),
        sleep_date date NOT NULL,
        totalMinutesAsleep float,
        totalSleepRecords integer,
        totalTimeInBed float
);

CREATE TABLE activity (
        subject_id character varying(10) NOT NULL references subject(id),
        activity_id integer NOT NULL,
        activity_parentId integer,
        calories float,
        description character varying(255),
        distance float,
        duration integer,
        has_start_time boolean,
        is_favorite boolean,
        log_id integer,
        name character varying(255),
        start_time timestamp,
        steps integer
);


CREATE TABLE activity_goal (
        subject_id character varying(10) NOT NULL references subject(id),
        calories_out integer,
        distance float,
        floors integer,
        steps integer
);

CREATE TABLE weight_goal (
   	subject_id character varying(10) NOT NULL references subject(id),
   	weight_goal_date date,
   	Start_weight integer,
   	Target_weight integer
);

CREATE TABLE fat_goal (
   	subject_id character varying(10) NOT NULL references subject(id),
   	Target_fat_percentage integer
);

CREATE TABLE weight_log (
   	subject_id character varying(10) NOT NULL references subject(id),
   	weight_log_date date,
   	Weight integer,
   	BMI float,
   	Source character varying(25)
);

CREATE TABLE user_attributes (
        subject_id character varying(10) NOT NULL references subject(id),
        city character varying(250),
        country character varying(25),
        distance_unit character varying(25),
        glucose_unit character varying(25),
        height integer,
        height_unit character varying(250),
        weight integer,
        weight_unit character varying(25)
);

CREATE TABLE heart_rate_intraday_data (
        subject_id character varying(10) NOT NULL references subject(id),
        log_time timestamp,
        heart_rate integer
);
