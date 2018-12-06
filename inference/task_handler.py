import boto3
import time
import json
import tenacity

retry = tenacity.retry(
  reraise=True, 
  stop=tenacity.stop_after_attempt(7), 
  wait=tenacity.wait_full_jitter(0.5, 60.0),
)

def make_residual_task_message(source_z, target_z, patch_bbox, mip):
  content = {
      "type": "residual_task",
      "source_z": source_z,
      "target_z": target_z,
      "patch_bbox": patch_bbox.serialize(),
      "mip": mip,
  }
  return json.dumps(content)

def make_prepare_task_message(z, patches, mip, start_z):
  content = {
      "type": "prepare_task",
      "z": z,
      "patches": [p.serialize() for p in patches],
      "mip": mip,
      "start_z": start_z,
  }
  return json.dumps(content)


def make_render_task_message(z, patches, mip, start_z):
  content = {
      "type": "render_task",
      "z": z,
      "patches": [p.serialize() for p in patches],
      "mip": mip,
      "start_z": start_z,
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


def make_downsample_task_message(z, patches, mip):
  content = {
      "type": "downsample_task",
      "z": z,
      "patches": [p.serialize() for p in patches],
      "mip": mip,
  }
  return json.dumps(content)

def make_copy_task_message(z, source, dest, patches, mip):
  content = {
      "type": "copy_task",
      "z": z,
      "source": source,
      "dest": dest,
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
    while(True):
        response = self.sqs.get_queue_attributes(QueueUrl=self.queue_url,
                                                 AttributeNames=attribute_names)
        Message_num = int(response['Attributes']['ApproximateNumberOfMessages'])
        if Message_num > threshold:
            print("Message number is", Message_num, "sleep")
            time.sleep(3)
        else: 
            self.sqs.send_message(QueueUrl=self.queue_url, 
                                  MessageBody=message_body)
            break
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

  def wait_until_ready(self):
    while not self.is_empty():
      time.sleep(1)

