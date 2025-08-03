function fakeSubscribe(){
  const input = document.getElementById('email');
  if(!input.checkValidity()){input.reportValidity();return false;}
  alert('Thanks! We\'ll be in touch soon.');
  return false;
}
document.getElementById('year').textContent = new Date().getFullYear();
