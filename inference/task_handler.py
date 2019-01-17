import boto3
import time
import json
import tenacity

retry = tenacity.retry(
  reraise=True, 
  stop=tenacity.stop_after_attempt(7), 
  wait=tenacity.wait_full_jitter(0.5, 60.0),
)

def make_residual_task_message(src_z, src_cv, tgt_z, tgt_cv, field_cv, patch_bbox, 
                               input_mip, output_mip):
  content = {                   
      "type": "residual_task",
      "src_z": src_z,
      "src_cv": src_cv.serialize(),
      "tgt_z": tgt_z,
      "tgt_cv": tgt_cv.serialize(),
      "field_cv": field_cv.serialize(),
      "patch_bbox": patch_bbox.serialize(),
      "input_mip": input_mip,
      "output_mip": output_mip,
  }
  return json.dumps(content)

def make_invert_field_task_message(z, src_cv, dst_cv, patch_bbox, mip, optimizer):
  content = {
      "type": "invert_task",
      "z": z,
      "src_cv": src_cv.serialize(),
      "dst_cv": dst_cv.serialize(),
      "patch_bbox": patch_bbox.serialize(),
      "mip": mip,
      "optimizer": optimizer,
  }
  return json.dumps(content)

def make_regularize_task_message(z_start, z_end, compose_start, patch_bbox, mip, sigma):
  content = {
      "type": "regularize_task",
      "z_start": z_start,
      "z_end": z_end,
      "compose_start": compose_start,
      "patch_bbox": patch_bbox.serialize(),
      "mip": mip,
      "sigma": sigma,
  }
  return json.dumps(content)

def make_vector_vote_task_message(z_range, read_F_cv, write_F_cv, patch_bbox, mip,
                                  inverse, T, negative_offsets, serial_operation):
  content = {
      "type": "vector_vote_task",
      "z_start": z_range[0],
      "z_end": z_range[-1],
      "read_F_cv": read_F_cv.serialize(),
      "write_F_cv": write_F_cv.serialize(),
      "patch_bbox": patch_bbox.serialize(),
      "mip": mip,
      "inverse": inverse,
      "T": T,
      "negative_offsets": negative_offsets,
      "serial_operation": serial_operation,
  }
  return json.dumps(content)


 #def make_vector_vote_task_message(z, read_F_cv, write_F_cv, patch_bbox, mip, inverse, T):
 #  content = {
 #      "type": "vector_vote_task",
 #      "z": z,
 #      #"z_end": z_range.stop,
 #      "read_F_cv": read_F_cv.serialize(),
 #      "write_F_cv": write_F_cv.serialize(),
 #      "patch_bbox": patch_bbox.serialize(),
 #      "mip": mip,
 #      "inverse": inverse,
 #      "T": T,
 #  }
 #  return json.dumps(content)


def make_prepare_task_message(z, patches, mip, start_z):
  content = {
      "type": "prepare_task",
      "z": z,
      "patches": [p.serialize() for p in patches],
      "mip": mip,
      "start_z": start_z,
  }
  return json.dumps(content)

def make_render_cv_task_message(z, field_cv, field_z, patches, mip, dst_cv, dst_z):
  content = {
      "type": "render_task_cv",
      "z": z,
      "field_cv": field_cv.serialize(),
      "field_z": field_z,
      "patches": [p.serialize() for p in patches],
      #"patches": patches.serialize(),
      "mip": mip,
      "dst_cv": dst_cv.serialize(),
      "dst_z": dst_z,
  }
  return json.dumps(content)

def make_render_task_message(z, field_cv, field_z, patches, mip, dst_cv, dst_z):
  content = {
      "type": "render_task",
      "z": z,
      "field_cv": field_cv.serialize(),
      "field_z": field_z,
      "patches": [p.serialize() for p in patches],
      #"patches": patches.serialize(),
      "mip": mip,
      "dst_cv": dst_cv.serialize(),
      "dst_z": dst_z,
  }
  return json.dumps(content)

def make_batch_render_message(z, field_cv, field_z, patches, mip, dst_cv,
                              dst_zm, batch):
  content = {
      "type": "batch_render_task",
      "z": z,
      "field_cv": field_cv.serialize(),
      "field_z": field_z,
      "patches": [p.serialize() for p in patches],
      #"patches": patches.serialize(),
      "mip": mip,
      "dst_cv": dst_cv.serialize(),
      "dst_z": dst_z,
      "batch": batch,
  }
  return json.dumps(content)

def make_render_low_mip_task_message(z, field_cv, field_z, patches, image_mip, vector_mip, dst_cv, dst_z):
  content = {
      "type": "render_task_low_mip",
      "z": z,
      "field_cv": field_cv.serialize(),
      "field_z": field_z,
      "patches": [p.serialize() for p in patches],
      "image_mip": image_mip,
      "vector_mip": vector_mip,
      "dst_cv": dst_cv.serialize(),
      "dst_z": dst_z,
  }
  return json.dumps(content)

def make_compose_task_message(z, patches, mip, start_z):
  content = {
      "type": "compose_task",
      "z": z,
      "patches": [p.serialize() for p in patches],
      "mip": mip,
      "start_z": start_z,
  }
  return json.dumps(content)


def make_downsample_task_message(cv, z, patches, mip):
  content = {
      "type": "downsample_task",
      "z": z,
      "cv": cv.serialize(),
      "patches": [p.serialize() for p in patches],
      #"patches": patches.serialize(),
      "mip": mip,
  }
  return json.dumps(content)

def make_copy_task_message(z, dst_cv, dst_z, patches, mip):
  content = {
      "type": "copy_task",
      "z": z,
      "dst_cv": dst_cv.serialize(),
      "dst_z": dst_z,
      "patches": [p.serialize() for p in patches],
      "mip": mip,
  }
  return json.dumps(content)

class TaskHandler:
  def __init__(self, queue_name):
    # Get the service resource
    self.sqs = boto3.client('sqs')

    self.queue_name = queue_name
    self.queue_url  = self.sqs.get_queue_url(QueueName=self.queue_name)["QueueUrl"]

  @retry
  def send_message(self, message_body):
    attribute_names = ['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
    threshold = 100000
    # while(True):
    #     response = self.sqs.get_queue_attributes(QueueUrl=self.queue_url,
    #                                              AttributeNames=attribute_names)
    #     Message_num = int(response['Attributes']['ApproximateNumberOfMessages'])
    #     if Message_num > threshold:
    #         print("Message number is", Message_num, "sleep")
    #         time.sleep(3)
    #     else: 
    #         self.sqs.send_message(QueueUrl=self.queue_url, 
    #                               MessageBody=message_body)
    #         break
    self.sqs.send_message(QueueUrl=self.queue_url, MessageBody=message_body)

  @retry
  def get_message(self, processing_time=90):
    response = self.sqs.receive_message(
      QueueUrl=self.queue_url,
      MaxNumberOfMessages=1,
      MessageAttributeNames=[
          'All'
      ],
      VisibilityTimeout=0
    )

    if 'Messages' in response.keys() and len(response['Messages']) > 0:
      message = response['Messages'][0]
      receipt_handle = message['ReceiptHandle']
      self.sqs.change_message_visibility(
              QueueUrl=self.queue_url,
              ReceiptHandle=receipt_handle,
              VisibilityTimeout=processing_time
      )
      return message
    else:
      return None
  
  @retry
  def delete_message(self, message):
    receipt_handle = message['ReceiptHandle']
    self.sqs.delete_message(
        QueueUrl=self.queue_url,
        ReceiptHandle=receipt_handle
    )

  @retry
  def is_empty(self):
    # hashtag hackerlife
    attribute_names = ['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
    for i in range(2):
      response = self.sqs.get_queue_attributes(QueueUrl=self.queue_url,
                                               AttributeNames=attribute_names)
      for a in attribute_names:
        if int(response['Attributes'][a]) > 0:
          return False
      time.sleep(5)
    return True
  
  def purge_queue(self):
    self.sqs.purge_queue(QueueUrl=self.queue_url)

  def wait_until_ready(self):
    while not self.is_empty():
      time.sleep(5)

