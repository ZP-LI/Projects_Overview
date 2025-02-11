// Auto-generated. Do not edit!

// (in-package mouse_controller.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class mouse_sensors {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.servo_pos_leg = null;
      this.servo_pos_aux = null;
      this.contact_sensors = null;
      this.imu_sensor = null;
    }
    else {
      if (initObj.hasOwnProperty('servo_pos_leg')) {
        this.servo_pos_leg = initObj.servo_pos_leg
      }
      else {
        this.servo_pos_leg = [];
      }
      if (initObj.hasOwnProperty('servo_pos_aux')) {
        this.servo_pos_aux = initObj.servo_pos_aux
      }
      else {
        this.servo_pos_aux = [];
      }
      if (initObj.hasOwnProperty('contact_sensors')) {
        this.contact_sensors = initObj.contact_sensors
      }
      else {
        this.contact_sensors = [];
      }
      if (initObj.hasOwnProperty('imu_sensor')) {
        this.imu_sensor = initObj.imu_sensor
      }
      else {
        this.imu_sensor = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type mouse_sensors
    // Serialize message field [servo_pos_leg]
    bufferOffset = _arraySerializer.float32(obj.servo_pos_leg, buffer, bufferOffset, null);
    // Serialize message field [servo_pos_aux]
    bufferOffset = _arraySerializer.float32(obj.servo_pos_aux, buffer, bufferOffset, null);
    // Serialize message field [contact_sensors]
    bufferOffset = _arraySerializer.float32(obj.contact_sensors, buffer, bufferOffset, null);
    // Serialize message field [imu_sensor]
    bufferOffset = _arraySerializer.float32(obj.imu_sensor, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type mouse_sensors
    let len;
    let data = new mouse_sensors(null);
    // Deserialize message field [servo_pos_leg]
    data.servo_pos_leg = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [servo_pos_aux]
    data.servo_pos_aux = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [contact_sensors]
    data.contact_sensors = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [imu_sensor]
    data.imu_sensor = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.servo_pos_leg.length;
    length += 4 * object.servo_pos_aux.length;
    length += 4 * object.contact_sensors.length;
    length += 4 * object.imu_sensor.length;
    return length + 16;
  }

  static datatype() {
    // Returns string type for a message object
    return 'mouse_controller/mouse_sensors';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '7a3639d3aca39f506211e62a51b3df2b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32[] servo_pos_leg
    float32[] servo_pos_aux
    float32[] contact_sensors
    float32[] imu_sensor
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new mouse_sensors(null);
    if (msg.servo_pos_leg !== undefined) {
      resolved.servo_pos_leg = msg.servo_pos_leg;
    }
    else {
      resolved.servo_pos_leg = []
    }

    if (msg.servo_pos_aux !== undefined) {
      resolved.servo_pos_aux = msg.servo_pos_aux;
    }
    else {
      resolved.servo_pos_aux = []
    }

    if (msg.contact_sensors !== undefined) {
      resolved.contact_sensors = msg.contact_sensors;
    }
    else {
      resolved.contact_sensors = []
    }

    if (msg.imu_sensor !== undefined) {
      resolved.imu_sensor = msg.imu_sensor;
    }
    else {
      resolved.imu_sensor = []
    }

    return resolved;
    }
};

module.exports = mouse_sensors;
