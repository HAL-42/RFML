
#include <ESP8266WiFi.h> //more about beacon frames https://mrncciew.com/2014/10/08/802-11-mgmt-beacon-frame/

#define kChannel 1
#define kIntervalms     1
#define kIntervalHexH   0x00
#define kIntervalHexL   0x01
char kSSID[] = "EsPap-SameMac";

extern "C" {
  #include "user_interface.h"
}

void setup() {
  delay(500);
  wifi_set_opmode(STATION_MODE);
  wifi_promiscuous_enable(1); 
}

void loop() {
  sendBeacon(kSSID); //sends beacon frames with the SSID 'test'
}

void sendBeacon(char* ssid) {
    // Set channel //
    wifi_set_channel(kChannel);

    uint8_t packet[128] = { 0x80, 0x00, //Frame Control 
                        0x00, 0x00, //Duration
                /*4*/   0xff, 0xff, 0xff, 0xff, 0xff, 0xff, //Destination address 
                /*10*/  0xec, 0x17, 0x2f, 0x2d, 0xb6, 0xb8, //Source address. Attention, use a truth mac address if possible, for fack address may illegal
                /*16*/  0xec, 0x17, 0x2f, 0x2d, 0xb6, 0xb8, //BSSID - same as the source address
                /*22*/  0xc0, 0x6c, //Seq-ctl
                //Frame body starts here
                /*24*/  0x83, 0x51, 0xf7, 0x8f, 0x0f, 0x00, 0x00, 0x00, //timestamp - the number of microseconds the AP has been active, can't been modified by program
                /*32*/  kIntervalHexL, kIntervalHexH, //Beacon interval
                /*34*/  0x11, 0x04, //Capability info
                /* SSID */
                /*36*/  0x00
                };

    int ssidLen = strlen(ssid);
    packet[37] = ssidLen;

    for(int i = 0; i < ssidLen; i++) {
      packet[38+i] = ssid[i];
    }

    uint8_t postSSID[13] = {0x01, 0x08, 0x82, 0x84, 0x8b, 0x96, 0x24, 0x30, 0x48, 0x6c, //supported rate
                        0x03, 0x01, 0x04 /*DSSS (Current Channel)*/ };

    for(int i = 0; i < 12; i++) {
      packet[38 + ssidLen + i] = postSSID[i];
    }

    packet[50 + ssidLen] = kChannel;

    // Randomize SRC MAC
//    packet[10] = packet[16] = random(256);
//    packet[11] = packet[17] = random(256);
//    packet[12] = packet[18] = random(256);
//    packet[13] = packet[19] = random(256);
//    packet[14] = packet[20] = random(256);
//    packet[15] = packet[21] = random(256);

    int packetSize = 51 + ssidLen;

//  while(1){
//    wifi_send_pkt_freedom(packet, packetSize, 0);
//    wifi_send_pkt_freedom(packet, packetSize, 0);
//    wifi_send_pkt_freedom(packet, packetSize, 0);
//    delay(kIntervalms);
//  }
//    wifi_send_pkt_freedom(packet, packetSize, 0);
//    wifi_send_pkt_freedom(packet, packetSize, 0);
//    wifi_send_pkt_freedom(packet, packetSize, 0);
//    delay(1);
    _sendBeacon(packet, packetSize);
}

// This function send out beacon repeatedly accroding to the interval setted
void _sendBeacon(uint8_t *packet, int packetSize){
  while(1){
    wifi_send_pkt_freedom(packet, packetSize, 0);
    delay(kIntervalms);
  }
}
