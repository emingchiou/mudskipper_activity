#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>
#include <Time.h>
#include <DS1307RTC.h>
#include <SD.h>
#include <SPI.h>
 
File f;

const unsigned long sec = 1000;
const unsigned long mins = 60 * sec;
const unsigned long hrs = 3600 * sec;

long period = 600;//unit = sec
long wait = 5*mins;//unit = sec

#define ONE_WIRE_BUS 7  //告訴 OneWire library DQ 接在那隻腳上
OneWire oneWire(ONE_WIRE_BUS); //建立onewire 物件
DallasTemperature DS18B20(&oneWire); //建立DS18B20物件

void setup() {
  Serial.begin(9600);
  pinMode(10, OUTPUT);
  DS18B20.begin();
    
  if (!SD.begin()){
    Serial.println(F("SD card x"));} 

  if(!RTC.get()){
    Serial.println(F("RTC x."));} 
}

void loop(void)
{ 
  long t = RTC.get()-28800;
  long a = 1621684800; //a>t+wait
  Serial.println(RTC.get()-28800);// for 校時
  while(RTC.get()){   
    delay(wait);
    while(RTC.get()){
      long t = RTC.get()-28800;
      if(t >= a){       
        break;
      }  
    }
    DS18B20.requestTemperatures(); //下指令開始轉換
    f = SD.open(F("temp.txt"), FILE_WRITE);
    if (!f) {
      Serial.println(F("error opening temp.txt"));} 
    Serial.println(DS18B20.getTempCByIndex(0));
    f.print(RTC.get()-28800);
    f.print(F(","));
    f.println(DS18B20.getTempCByIndex(0));
    f.close(); 
    
    a += period;
  }
}
