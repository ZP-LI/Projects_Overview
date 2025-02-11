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

class desired_cmd {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.vel = null;
      this.turn_rate = null;
      this.buttons = null;
    }
    else {
      if (initObj.hasOwnProperty('vel')) {
        this.vel = initObj.vel
      }
      else {
        this.vel = 0.0;
      }
      if (initObj.hasOwnProperty('turn_rate')) {
        this.turn_rate = initObj.turn_rate
      }
      else {
        this.turn_rate = 0.0;
      }
      if (initObj.hasOwnProperty('buttons')) {
        this.buttons = initObj.buttons
      }
      else {
        this.buttons = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type desired_cmd
    // Serialize message field [vel]
    bufferOffset = _serializer.float32(obj.vel, buffer, bufferOffset);
    // Serialize message field [turn_rate]
    bufferOffset = _serializer.float32(obj.turn_rate, buffer, bufferOffset);
    // Serialize message field [buttons]
    bufferOffset = _arraySerializer.int32(obj.buttons, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type desired_cmd
    let len;
    let data = new desired_cmd(null);
    // Deserialize message field [vel]
    data.vel = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [turn_rate]
    data.turn_rate = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [buttons]
    data.buttons = _arrayDeserializer.int32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.buttons.length;
    return length + 12;
  }

  static datatype() {
    // Returns string type for a message object
    return 'mouse_controller/desired_cmd';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '6ca41f915100349b7a4af3c164f6abe2';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 vel
    float32 turn_rate
    int32[] buttons
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new desired_cmd(null);
    if (msg.vel !== undefined) {
      resolved.vel = msg.vel;
    }
    else {
      resolved.vel = 0.0
    }

    if (msg.turn_rate !== undefined) {
      resolved.turn_rate = msg.turn_rate;
    }
    else {
      resolved.turn_rate = 0.0
    }

    if (msg.buttons !== undefined) {
      resolved.buttons = msg.buttons;
    }
    else {
      resolved.buttons = []
    }

    return resolved;
    }
};

module.exports = desired_cmd;
