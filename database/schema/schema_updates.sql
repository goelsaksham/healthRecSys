alter table public.raw_device_data
add constraint raw_device_data_subject_id_fkey FOREIGN KEY (subject_id)
      REFERENCES public.subject (id) MATCH SIMPLE
      ON UPDATE NO ACTION ON DELETE NO ACTION;

alter table public.raw_device_data
   alter column subject_id type character varying(10);

drop table sleep_cycles;
  
CREATE TABLE sleep_cycles (
        subject_id character varying(10) NOT NULL references subject(id),
		dateOfSleep date, 
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

alter table activity_intraday_data drop column activity_level_desc;

alter table activity_type_summary rename column value to calories;

select * from activity_type_summary order by date;